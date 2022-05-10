import os
from collections import defaultdict

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from timm.data.transforms_factory import create_transform 
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset

def print_dataset_summary(dataset):
    """
    Print statistics about the label distributions on given
    pytorch dataset
    :param dataset: A pytorch dataset
    """
    label2count = defaultdict(lambda : 0)
    total = 0
    for datapoint in dataset:
        label2count[datapoint[1]] += 1
        total += 1
    print("=========Dataset summary=========")
    for label, count_i in label2count.items():
        print(f"Label {label}, count {count_i}, "
              f"percentage {count_i / total}")


def plot_dataset_image(dataset , index):
    plt.imshow(dataset[index][0].transpose(0, 2).transpose(0, 1))
    plt.show()

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

class CustomDataset(Dataset):
    def __init__(self, idxlist, dataloader_):
        self.idx_list = idxlist
        self.dataloader = dataloader_

    def __getitem__(self, idx: int):
            return self.dataloader[self.idx_list[idx]]

    def __len__(self):
        return len(self.idx_list)

def train_val_stratified_breed_split(train_dataloader):
    seed_everything(0)
    # input: trainval dataloader
    # Taka 20 samples out of each class for validation
    # use everything else for training
    # returns: training and validation indices
    j = 20
    current_class=0
    train_idx = []
    validation_idx = []
    idx = 0
    for dat, lab in train_dataloader:

        if j == 0:
            current_class = current_class +1
            j = 20
        if lab == current_class and j!=0:
            # validation_data.append(dat)
            # validation_labels.append(lab)
            validation_idx.append(idx)
            j = j - 1
        else:
            # train_data.append(dat)
            # train_labels.append(lab)
            train_idx.append(idx)
        idx = idx + 1

    return CustomDataset(train_idx, train_dataloader), CustomDataset(
        validation_idx, train_dataloader)

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
        if str(i.item() + 1) in cat_dog_dict['cat']:
            cat_dog_labels.append(0)
        elif str(i.item() + 1) in cat_dog_dict['dog']:
            cat_dog_labels.append(1)
        else:
            print(i.item())
            raise ValueError

    return torch.LongTensor(cat_dog_labels)



def output_jpg_dir_of_training_data(output_path):
    os.mkdir(output_path)
    train_ds, test_ds = download_dataset(0)
    for i, (image, label) in enumerate(train_ds):
        save_image(image, os.path.join(output_path, f"image-{label}-{i}.jpg"))


def download_dataset(augmentation = False):
    '''
    Parameters:
    augumentation : do you to perform data augumentation like cropping etc.., set to true  

    Returns:
    train and test OxfordIIITPet dataset
    '''
    img_size = 255
    if augmentation:
        train_transform = create_transform(img_size, is_training = True)
    else:
        train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size))
    ])
    training_data = datasets.OxfordIIITPet(root="data", split="trainval", download=True, transform=train_transform)
    test_data = datasets.OxfordIIITPet(root="data", split="test", download=True, transform=test_transform)

    return training_data, test_data


# note that in the data/annotations/list.txt has filename | 1:37 Class ids | 1:Cat 2:Dog | 1-25:Cat 1:12:Dog
# so function given an id get whether cat or dog,
# creating a dictionary with {'cat':[list of ids], 'dog':[list of ids]}, computed in the beginning first and then retrive later.
def create_cat_dog_dict():
    '''
    Returns:
    {'cat':[list of ids], 'dog':[list of ids]}
    '''
    cat_list = []
    dog_list = []
    list_file_path = "data/oxford-iiit-pet/annotations/list.txt"
    a_file = open(list_file_path)
    lines = a_file.readlines()[6:]
    for line in lines:
        id = line.rstrip().split()[1]
        if id in cat_list or id in dog_list:
            continue
        cat_dog = line.rstrip().split()[2]
        if cat_dog == '1':
            cat_list.append(id)
        else:
            dog_list.append(id)
    a_file.close()
    cat_dog_dict = {"cat": cat_list, "dog": dog_list}
    return cat_dog_dict
