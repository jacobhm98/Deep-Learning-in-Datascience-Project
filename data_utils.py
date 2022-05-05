import os
import shutil
from pathlib import Path

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms


def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_val_split(train_dataset):
    seed_everything(0)
    num_train_images = len(train_dataset)
    val_size = int(0.1 * num_train_images)
    train_size = int(0.9 * num_train_images)
    return torch.utils.data.random_split(train_dataset, lengths=[train_size,
                                                                 val_size])


def gen_cat_dog_label(cat_dog_dict, labels):
    '''
    Parameters :
    cat_dog_dict : {'cat':[cat_id_list], 'dog':[dog_id_list]}
    labels : tensor of ids
    Returns :
    cat_dog_labels : tensor of cats(0) and dogs(1)
    '''
    cat_dog_labels = []
    for i in labels:
        if i in cat_dog_dict['cat']:
            cat_dog_labels.append(0)
        else:
            cat_dog_labels.append(1)

    return torch.LongTensor(cat_dog_labels)

from torchvision.utils import save_image

def output_jpg_dir_of_training_data(output_path):
    os.mkdir(output_path)
    train_ds, test_ds = download_dataset(0)
    for i, (image, label) in enumerate(train_ds):
        save_image(image, os.path.join(output_path, f"image-{label}-{i}.jpg"))


def download_dataset(batch_size):
    '''
    Parameters:
    batch_size: Batch size (needed in dataloader)
    Returns:
    train and test OxfordIIITPet dataset
    '''
    img_size = 255
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size))
    ])
    training_data = datasets.OxfordIIITPet(root="data", split="trainval", download=True, transform=transform)
    test_data = datasets.OxfordIIITPet(root="data", split="test", download=True, transform=transform)

    return training_data, test_data

