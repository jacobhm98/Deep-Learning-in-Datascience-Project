import os
from collections import defaultdict
from typing import List

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import pandas as pd


def print_dataset_summary(dataset):
    """
    Print statistics about the label distributions on given
    pytorch dataset
    :param dataset: A pytorch dataset
    """
    label2count = defaultdict(lambda: 0)
    total = 0
    for datapoint in dataset:
        label2count[datapoint[1]] += 1
        total += 1
    print("=========Dataset summary=========")
    for label, count_i in label2count.items():
        print(f"Label {label}, count {count_i}, "
              f"percentage {count_i / total}")


def demo_transformations(dataset):
    idxs = list(range(len(dataset)))
    indexes = np.random.choice(idxs, size=3)
    fig, axs = plt.subplots(3, 5)
    for i in range(3):
        for j in range(5):
            axs[i][j].imshow(
                dataset[indexes[i]][0].transpose(0, 2).transpose(0, 1)
            )
    plt.show()


def plot_dataset_image(dataset, index):
    plt.imshow()
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
    def __init__(self, idxlist, dataloader_, transform):
        self.idx_list = idxlist
        self.dataloader = dataloader_
        self.transform = transform

    def __getitem__(self, idx: int):
        val = self.dataloader[self.idx_list[idx]]
        return self.transform(val[0]), val[1]

    def __len__(self):
        return len(self.idx_list)


def train_val_stratified_breed_split(train_dataloader, train_transform,
                                     test_transform, num_ex = 80):
    seed_everything(0)
    # input: trainval dataloader
    # Taka num_ex samples out of each class for train_set (default = 20)
    # use everything else for training
    # returns: training and validation indices
    j = num_ex
    current_class = 0
    train_idx = []
    validation_idx = []

    count_per_label = defaultdict(lambda : 0)

    for i, (dat, lab) in enumerate(train_dataloader):
        if count_per_label[lab] < num_ex:
            train_idx.append(i)
            count_per_label[lab] += 1
        else:
            validation_idx.append(i)

    return CustomDataset(train_idx, train_dataloader, train_transform), CustomDataset(
        validation_idx, train_dataloader, test_transform)

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


def download_dataset(augmentation=False, in_memory=False,
                     train_transforms=None, num_train_examples=80):
    '''
    Parameters:
    augumentation : do you to perform data augumentation like cropping etc.., set to true

    Returns:
    train and test OxfordIIITPet dataset
    '''
    print("Downloading datasets")
    img_size = 255
    all_data = datasets.OxfordIIITPet(root="data", split="trainval",
                                      download=True)

    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
    ])

    import os
    if os.path.exists("data/petmean"):
        with open("data/petmean", 'rb') as f:
            mean = torch.load(f)
        with open("data/petstd", 'rb') as f:
            std = torch.load(f)
    else:

        print("Calculating mean, stds on train")
        concatenated_train_val = torch.cat(
            [base_transforms(x[0]).unsqueeze(-1) for x in all_data], dim=-1)
        reshaped_train_val = concatenated_train_val.reshape(3, -1)

        mean = reshaped_train_val.mean(1)
        std = reshaped_train_val.std(1)

        with open("data/petmean", 'wb') as f:
            torch.save(mean, f)
        with open("data/petstd", 'wb') as f:
            torch.save(std, f)
    norm_tf = transforms.Normalize(mean=mean,
                                   std=std)

    if augmentation:
        train_transform = transforms.Compose([
            train_transforms,
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            norm_tf
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            norm_tf
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            norm_tf
        ])
        test_transform = train_transform

    test_data = datasets.OxfordIIITPet(root="data", split="test",
                                       download=True, transform=test_transform)

    if in_memory:
        print("Inmemorizing...")
        all_data = inmemorize_dataset(all_data)
        test_data = inmemorize_dataset(test_data)

    print("Splitting to train, val, test")
    train_data, val_data = train_val_stratified_breed_split(all_data,
                                                            train_transform,
                                                            test_transform,
                                                            num_train_examples)


    demo_transformations(train_data)

    return train_data, val_data, test_data


class ListDataset():
    def __init__(self, l: List):
        self.l = l

    def __getitem__(self, item):
        return self.l[item]

    def __len__(self):
        return len(self.l)


def inmemorize_dataset(dataset):
    examples = [dataset[i] for i in range(len(dataset))]
    return ListDataset(examples)


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


def create_train_plot(train_metrics_file):
    train_metrics = pd.read_csv(train_metrics_file)
    train_metrics.plot(x=0, y=1, kind="line", xlabel="batches trained")
    plt.show()


def create_val_plot(val_metrics_file):
    val_metrics = pd.read_csv(val_metrics_file)
    val_metrics.plot(x=0, y=1, kind="line", xlabel="epochs")
    plt.show()
