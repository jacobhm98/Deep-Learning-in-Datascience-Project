#!/bin/bash
mkdir ../../gans_training/dataset
python ../../stylegan2-ada-pytorch/dataset_tool.py --source ../../gans_training/images/ --dest ../../gans_training/dataset/
