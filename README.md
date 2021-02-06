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
