import torch
import torch.nn as nn
from torch.nn import init
import functools
from torchvision import transforms, utils
from PIL import Image
import os
import io
import logging
import json
import requests
import boto3
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _s3_path_split(s3_path):
    s3_path = s3_path.strip()
    if not s3_path.startswith("s3://"):
        raise ValueError(
            f"s3_path is expected to start with 's3://', but was {s3_path}"
        )
    bucket_key = s3_path[len("s3://") :]
    bucket_name, key = bucket_key.split("/", 1)
    return bucket_name, key


def image_loader(s3_resource, image_url, image_size=None, max_width=800, max_height=600):
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
            image_size = (round(width * scale_factor), round(height * scale_factor))
    if image_size is None:
        loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        loader = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image


def get_result_image_as_bytes(result_img):
    result_img = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))(result_img.squeeze())
    result_img_as_bytes = io.BytesIO()
    utils.save_image(result_img, result_img_as_bytes, format='jpeg')
    result_img_as_bytes.seek(0)
    return result_img_as_bytes


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Model(nn.Module):
    def __init__(self, model_dir, style_name):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.netG = ResnetGenerator(norm_layer=norm_layer)


        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        self.netG.to(self.device)
        logger.info('initialize network with normal')
        self.netG.apply(init_func)  # apply the initialization function <init_func>

        load_path = os.path.join(model_dir, style_name + '.pth')
        logger.info('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.netG, key.split('.'))
        self.netG.load_state_dict(state_dict)

        logger.info('---------- Networks initialized -------------')
        num_params = 0
        for param in self.netG.parameters():
            num_params += param.numel()
        logger.info('[Network netG] Total number of parameters : %.3f M' % (num_params / 1e6))
        logger.info('-----------------------------------------------')

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input.to(self.device)

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def eval(self):
        """Make model eval mode during test time"""
        self.netG.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()

    def get_output(self):
        """Return output image. """
        return self.fake


def model_fn(model_dir):
    return model_dir


def transform_fn(model_dir, data, content_type='application/json', output_content_type='application/json'): 
    if output_content_type != 'application/json':
        raise ValueError(f'Unsupported output content type: {output_content_type}')

    if content_type == 'application/json':
        data = json.loads(data)
    elif content_type == 'text/csv':
        df = pd.read_csv(io.StringIO(data), header=None, names=['content_url', 'style_name', 'result_url'], skipinitialspace=True)
        data = df.loc[0, :].to_dict()
        data = {key:value for key, value in data.items() if not pd.isnull(value)}
    else:
        raise ValueError(f'Unsupported content type: {content_type}')
    logger.info(data)
        
    s3_resource = boto3.Session().resource('s3')    
    content_img = image_loader(s3_resource, data['content_url'])
       
    model = Model(model_dir, data['style_name'])
    model.eval()
    model.set_input(content_img)
    model.test()
    result_img_as_bytes = get_result_image_as_bytes(model.get_output())
    bucket_name, key = _s3_path_split(data['result_url'])
    s3_resource.Bucket(bucket_name).Object(key).upload_fileobj(result_img_as_bytes)
    response_body = json.dumps({'result_url': data['result_url']})
    return response_body, output_content_type
