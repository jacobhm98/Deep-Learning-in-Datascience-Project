#!/bin/bash

cd ../..
mkdir gans_training
mkdir gans_training/experiments
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
pip install ninja
unzip images.zip -d gans_training/

