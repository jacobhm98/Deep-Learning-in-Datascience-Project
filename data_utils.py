import torch
import numpy as np


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
