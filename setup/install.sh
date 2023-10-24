#!/bin/bash
conda create --name mbrl python=3.9.7
conda activate mbrl
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
pip install TorchDiffEqPack
pip install imageio-ffmpeg
conda install -n mbrl ipykernel --update-deps --force-reinstall
sudo apt-get install xvfb
sudo apt-get install python-opengl