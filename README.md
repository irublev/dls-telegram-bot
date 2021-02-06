# dls-telegram-bot


### General info

This repository implements [Telegram bot](https://t.me/DLSTelegramBot) that performs neural style transfer of some content image

* either using some style image via the [Neural-Style algorithm](https://arxiv.org/abs/1508.06576) developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge
* or by several pretrained [CycleGAN models](https://arxiv.org/pdf/1703.10593.pdf) developed by Jun-Yan Zhu, Taesung Park, Phillip Isola and Alexei A. Efros

The latter CycleGAN models allow to transfer any content image to one of four predefined styles:

* Monet
* Van Gogh
* Cezanne
* Ukiyo-e

The code for Neural-Style algorithm is based on the paper [Neural Transfer using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) by Alexis Jacq, but significantly refactored using [PyTorch Lightining](https://www.pytorchlightning.ai/), besides some minor changes are done. What concerns pretrained CycleGAN models, the code implementing the inference for their generators is extracted from [CycleGAN and pix2pix in PyTorch](https://github.com/mashyko/pytorch-CycleGAN-and-pix2pix) with the code written by Jun-Yan Zhu and Taesung Park.

This [Telegram bot](https://t.me/DLSTelegramBot) was deployed in Heroku while all the models are designed to be deployed in AWS SageMaker.

**Imortant note** [Telegram bot](https://t.me/DLSTelegramBot) uses models deployed via SageMaker on paid AWS instances, please communicate with the author before using this bot concerning launching necessary AWS resources.

### Interface

[Telegram bot](https://t.me/DLSTelegramBot) implements three commands:

* /start to start new style transfer request
* /cancel to cancel current style transfer request
* /help for displaying help

At start of a style transfer request the bot asks to upload an image with content to be processed, this image may be uploaded both with and without compression. After the content image is uploaded, the bot asks either to choose the name of one of four predefined styles (see above) from the keyboard or to upload an image with style to apply. After this the chosen style is applied to the content image and the result is sent automatically to the chat. At any instant it is possible to cancel current request and to start a new one. If case of any errors the bot displays the info what is wrong and what is expected to be done by a user.

### Architecture

[Telegram bot](https://t.me/DLSTelegramBot) is implemented via [AIOgram framework](https://github.com/aiogram/aiogram) allowing to process all user messages asynchronously. Besides, it implements a finite-state machine, but not via standard AIOGram [FSM](https://docs.aiogram.dev/en/latest/examples/finite_state_machine_example.html) that either allows to use very unreliable memory storage or requires to deploy additionally some database such as Redis or MongoDB. This bot stores its data for each chat (and states fully determined by the contents of this data) in AWS S3, moreover, results are also stored in AWS S3 which is rather natural for SageMaker. By this way the persistency is easily achieved.

For these purposes we have a special S3 bucket, in this bucket for each chat some subfolder is created where the data is stored either as text files or as images. What concerns text files, they are files containing URLs with content or style images or style names as well as a special file with result counter necessary to avoid conflicts between already cancelled and yet processing requests.

When [Telegram bot](https://t.me/DLSTelegramBot) obtains all the necessary data for style transfer, it invokes one of AWS SageMaker endpoints for models deployed in AWS containers on special AWS instances. Each of these models processes the inputs and saves the resulting image in AWS S3 (the path for this image is given as one of inputs). After invoking endpoints the bot takes the resulting image from S3 and sends its to the corresponding chat.

### Code structure

The code for [Telegram bot](https://t.me/DLSTelegramBot) itself is in **bot** folder. The code for the models is in **sagemaker** folder (**sagemaker/nst** and **sagemaker/cyclegan**). The models are deployed in AWS SageMaker via **deploy-models.ipynb** notebook (all that is necessary for this Jupyter notebook is to install sagemaker package as pointed in **requirements.txt**). All the necessary prerequisites for deploying models are pointed in **deploy-models.ipynb**.
