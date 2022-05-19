import os
from PIL import image
path = ../../"gans_training/images"

for item in os.listdir(path):
    im = Image.open(path + '/' +  item)
    if im.mode != "RGB" :
        im = im.convert("RGB")
    imResize = im.resize((128,128))
    imResize.save(path + '/' +  item )
