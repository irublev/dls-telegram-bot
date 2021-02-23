import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms, utils
import torch.optim as optim
from PIL import Image
import os
import io
import pytorch_lightning as pl
import warnings
import logging
from copy import deepcopy
import json
import requests
import boto3
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _s3_path_split(s3_path):
    '''
    Split S3 path on bucket name and key
    '''
    s3_path = s3_path.strip()
    if not s3_path.startswith("s3://"):
        raise ValueError(
            f"s3_path is expected to start with 's3://', but was {s3_path}"
        )
    bucket_key = s3_path[len("s3://") :]
    bucket_name, key = bucket_key.split("/", 1)
    return bucket_name, key


def image_loader(s3_resource, image_url, device=None, image_size=None, max_width=800, max_height=600):
    '''
    Retrieve image as bytes either by URL or by S3 path, resize it either to given image size
    or to make it fit to maximal width and height (the latter is just to avoid problems with memory),
    and return image as a PyTorch tensor
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading image {image_url} as bytes...")
    image_url = image_url.strip()
    if image_url.startswith('s3://'):
        bucket_name, key = _s3_path_split(image_url)
        image_obj = s3_resource.Object(bucket_name=bucket_name, key=key)
        image_as_bytes = io.BytesIO(image_obj.get()['Body'].read())
    else:
        image_as_bytes = requests.get(image_url, stream=True).raw
    logger.info(f"Loading image {image_url} as bytes: done")
    image = Image.open(image_as_bytes).convert('RGB')
    if image_size is None and (max_width is not None or max_height is not None):
        width, height = image.size
        x_scale_factor = 1.0
        y_scale_factor = 1.0
        if max_width is not None and width > max_width:
            x_scale_factor = max_width / width
        if max_height is not None and height > max_height:
            y_scale_factor = max_height / height
        scale_factor = min(x_scale_factor, y_scale_factor)
        if scale_factor < 1.0:
            image_size = (round(height * scale_factor), round(width * scale_factor))
    if image_size is None:
        loader = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        loader = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class ContentLoss(nn.Module):
    '''
    module calculating content loss
    '''

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = nn.Parameter(target.detach(), requires_grad=False)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    '''
    module calculating style loss
    '''

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = nn.Parameter(self.__gram_matrix(target_feature).detach(), requires_grad=False)

    def forward(self, input):
        self.loss = F.mse_loss(self.__gram_matrix(input), self.target)
        return input

    @staticmethod
    def __gram_matrix(input):
        '''
        Calculate Gram matrix
        '''
        b, c, h, w = input.size()
        features = input.view(b * c, h * w)
        G = torch.mm(features, features.t())

        return G.div(input.numel())


# this class is necessary just because torchvision.transformers.Normalize
# implemented in the versions included in PyTorch Docker containers (provided
# by SageMaker with PyTorch framework 1.5.0) is not inherited from
# torch.nn.Module (PyTorch Docker containers for framework 1.6.0 seem to be
# not fully compatible with latest version of sagemaker package)
class ModuleNormalize(nn.Module):
    '''
    module normalizing image, analogous to torchvision.transformers.Normalize
    '''

    def __init__(self, mean, std, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(ModuleNormalize, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, image_tensor):
        return (image_tensor - self.mean) / self.std

	
class Model(nn.Module):
    '''
    NST model
    '''

    def __init__(self, cnn, content_img, style_img, content_conv_layer_ind_set=None, style_conv_layer_ind_set=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(Model, self).__init__()
        # default indices of convolution layers after which layers calculating content and style losses are to be inserted,
        # we are using rather deep layers following https://arxiv.org/abs/1508.06576
        if content_conv_layer_ind_set is None:
            content_conv_layer_ind_set = {13}  # indices of convolution layers after which to insert content loss
        if style_conv_layer_ind_set is None:
            style_conv_layer_ind_set = {1, 3, 5, 9, 13}  # indices of convolution layers after which to insert style loss
        cnn = deepcopy(cnn)
        for param in cnn.parameters():
            param.requires_grad = False

        # normalize image at first
        normalization = ModuleNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device=device)
        model = nn.Sequential(normalization)

        if not isinstance(content_conv_layer_ind_set, set):
            content_conv_layer_ind_set = set(content_conv_layer_ind_set)
        if not isinstance(style_conv_layer_ind_set, set):
            style_conv_layer_ind_set = set(style_conv_layer_ind_set)
        # index of convolution layer after last loss inserted, we cut off layers of original CNN model
        # after this layer
        max_conv_layer = max(max(content_conv_layer_ind_set), max(style_conv_layer_ind_set))
        ind_conv_layer = 0
        content_losses = []
        style_losses = []
        for ind_layer, layer in enumerate(cnn.children()):
            if isinstance(layer, nn.Conv2d):
                ind_conv_layer += 1
                name = 'conv_{}'.format(ind_layer)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(ind_layer)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(ind_layer)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ind_layer)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            model.add_module(name, layer)

            if isinstance(layer, nn.Conv2d):
                if ind_conv_layer in content_conv_layer_ind_set:
                    # add content loss
                    target = model(content_img).detach()
                    content_loss = ContentLoss(target)
                    model.add_module("content_loss_{}".format(ind_layer), content_loss)
                    content_losses.append(content_loss)
                if ind_conv_layer in style_conv_layer_ind_set:
                    # add style loss
                    target_feature = model(style_img).detach()
                    style_loss = StyleLoss(target_feature)
                    model.add_module("style_loss_{}".format(ind_layer), style_loss)
                    style_losses.append(style_loss)
                if ind_conv_layer == max_conv_layer:
                    break
        self.model = model
        self.content_img = nn.Parameter(content_img.clone())
        self.content_losses = nn.ModuleList(content_losses)
        self.style_losses = nn.ModuleList(style_losses)

    def forward(self):
        return self.model(self.content_img)


class NeuralStyleTransfer(pl.LightningModule):
    '''
    PyTorch Lightining module implementing Neural-Style algorithm (https://arxiv.org/abs/1508.06576)

    It is necessary to note that we are using several training steps per each epoch, this is done
    just to make an overhed when epoch changes less (this overhead is due PyTorch Lightning implementation)
    and so to make this implementation work faster
    '''

    def __init__(self, model, optimizer, content_weight=1e-5, style_weight=1e+4, n_print_steps=100):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.n_print_steps = n_print_steps

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, _, __):
        self.model.forward()
        # calculate total loss (being a sum of all content and style losses) at each training step
        loss = 0
        for content_loss in self.model.content_losses:
            loss += self.content_weight * content_loss.loss
        for style_loss in self.model.style_losses:
            loss += self.style_weight * style_loss.loss
        global_step = self.global_step
        current_epoch = self.current_epoch
        if global_step % self.n_print_steps == 0:
            logger.info(f"epoch: {current_epoch}, global_step: {global_step}, total loss: {loss}")
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True
        )
        return loss

    def optimizer_step(self, _, __, optimizer, ___, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.model.content_img.data.clamp_(0, 1)


def train_and_get_result(lmodule, max_epochs, n_steps_in_epoch=100, patience=3):

    warnings.filterwarnings('ignore')

    # artificial dataloader, it is necessary just to make severals steps per each epoch to make PyTorch Lightning overhead less
    dataloader = DataLoader(TensorDataset(torch.ones(n_steps_in_epoch)), batch_size=1)

    callbacks = []
    if patience is not None:
        callbacks.append(pl.callbacks.EarlyStopping(monitor='train_loss_epoch', patience=patience, verbose=True, mode='min'))
    callbacks.append(pl.callbacks.ModelCheckpoint(monitor='train_loss_epoch', mode='min', save_top_k=3, period=1, filename='nst-{epoch:02d}-{train_loss_epoch:.6f}'))

    if torch.cuda.is_available():
        trainer = pl.Trainer(deterministic=True, gpus=-1, auto_select_gpus=True, callbacks=callbacks, max_epochs=max_epochs)
    else:
        trainer = pl.Trainer(deterministic=True, gpus=0, callbacks=callbacks, max_epochs=max_epochs)

    # fit the model
    trainer.fit(lmodule, dataloader)

    # load the best model from the checkpoint
    best_model = pl.utilities.cloud_io.load(trainer.checkpoint_callback.best_model_path, map_location=lambda storage, loc: storage)['state_dict']

    # return resulting image
    return best_model['model.content_img']


def model_fn(model_dir):
    '''
    Callback used by SageMaker to create a model
    
    We are using this callback to load pretrained VGG19 model (its state is taken from S3)
    which is to be modified in class Model above
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg19(pretrained=False).features
    with open(os.path.join(model_dir, 'vgg19_state_dict.model'), 'rb') as f:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(f))
        else:
            model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    return model.to(device).eval()


def transform_fn(model, data, content_type='application/json', output_content_type='application/json'): 
    '''
    Callback used by SageMaker to retrieve inputs, pass them to a model, and return results

    Inputs are given as JSON (or CSV file with the same fields, the latter is used for
    SageMaker batch transform jobs) with the following fields:

       content_url: URL or S3 path for content image
       style_url: URL or S3 path for style image
       result_url: S3 path where the result image is to be saved
       max_epochs: maximal number of epoches to proceed
       n_steps_in_epoch: number of steps per each epoch (see above NeuralStyleTransfer for details)

    Output is JSON with the single field result_url containing S3 path where the result image is saved
    '''

    if output_content_type != 'application/json':
        raise ValueError(f'Unsupported output content type: {output_content_type}')

    if content_type == 'application/json':
        data = json.loads(data)
    elif content_type == 'text/csv':
        df = pd.read_csv(io.StringIO(data), header=None, 
            names=['content_url', 'style_url', 'result_url', 'max_epochs', 'n_steps_in_epoch', 'lr', 'patience', 'content_weight', 'style_weight'],
            skipinitialspace=True)
        data = df.loc[0, :].to_dict()
        data = {key:value for key, value in data.items() if not pd.isnull(value)}
    else:
        raise ValueError(f'Unsupported content type: {content_type}')
    logger.info(data)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s3_resource = boto3.Session().resource('s3')    
    # retrieve content image
    content_img = image_loader(s3_resource, data['content_url'], device=device)
    image_size = tuple(content_img.shape[2:])
    # retrieve style image and resize it to the size of the content image
    style_img = image_loader(s3_resource, data['style_url'], device=device, image_size=image_size)
       

    if 'max_epochs' in data:
        max_epochs = data['max_epochs']
    else:
        max_epochs = 10
    if 'n_steps_in_epoch' in data:
        n_steps_in_epoch = data['n_steps_in_epoch']
    else:
        n_steps_in_epoch = 100
    if 'lr' in data:
        lr = data['lr']
    else:
        lr = 0.01
    if 'patience' in data:
        patience = data['patience']
    else:
        patience = 3
    if 'content_weight' in data:
        content_weight = data['content_weight']
    else:
        content_weight = 1e-5
    if 'style_weight' in data:
        style_weight = data['style_weight']
    else:
        style_weight = 1e+4


    # create model and module implementing NST
    model = Model(model, content_img, style_img).to(device)
    lmodule = NeuralStyleTransfer(model, 
        torch.optim.Adam([model.content_img], lr=lr),
        content_weight=content_weight, style_weight=style_weight, n_print_steps=n_steps_in_epoch)

    # launch the algo and get the result image tensor
    result_img = train_and_get_result(lmodule, max_epochs=max_epochs, n_steps_in_epoch=n_steps_in_epoch, patience=patience)
    # convert result image tensor to bytes and put this to S3
    bucket_name, key = _s3_path_split(data['result_url'])
    result_img_as_bytes = io.BytesIO()
    utils.save_image(result_img.squeeze(), result_img_as_bytes, format='jpeg')
    result_img_as_bytes.seek(0)
    s3_resource.Bucket(bucket_name).Object(key).upload_fileobj(result_img_as_bytes)
    # return S3 path with the resulting image as an output
    response_body = json.dumps({'result_url': data['result_url']})
    return response_body, output_content_type
