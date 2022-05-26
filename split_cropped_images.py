import os
from PIL import Image
import random
path = "data/cropped_images"

for item in os.listdir(path):
    im = Image.open(path + '/' +  item)
    num = random.randint(1, 10)
    if num == 1:
        im.save(path + "_test/" + item)
    else:
        im.save(path + '_train/' +  item )
