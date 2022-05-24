#!/bin/bash

for i in {0..36}
do
	mkdir "/content/drive/MyDrive/gans_training/generated_images/class_$i"
	python /content/stylegan2-ada-pytorch/generate.py --network=/content/drive/MyDrive/network-snapshot-004200.pkl --outdir="/content/drive/MyDrive/gans_training/generated_images/class_$i" --seeds="1-$1" --class="$i"
done
