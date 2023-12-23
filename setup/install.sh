#!/bin/bash

# Set up conda env:
conda create --name mbrl python=3.9.7 -y
conda activate mbrl

# Pytorch-related:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Alternatively
# pip install torch torchvision torchaudio

# Gym v0.21.0 requires specific setuptools/wheel to be installed successfully.
pip install "setuptools==65.5.0"
pip install "wheel==0.38.4"
pip install "gym==0.21.0"
pip install --upgrade setuptools wheel

# The rest of pip requirements:
pip install -r ./setup/requirements.txt
pip install TorchDiffEqPack

# FFMpeg-related:
conda install -c conda-forge ffmpeg -y
pip install imageio-ffmpeg

# ipykernel:
conda install -n mbrl ipykernel --update-deps --force-reinstall -y

# System packages:
sudo apt-get install xvfb
sudo apt-get install python-opengl

# NOTE:
# If `torch.cuda.is_available()` returns False, reinstalling pytorch at the end of this process may help.
