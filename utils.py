import torch
import numpy as np

def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_val_split(train_dataset):
    seed_everything(0)
    num_train_images = len(train_dataset)
    val_percentage = 0.1
    val_size = int(0.1 * num_train_images)
    train_size = int(0.9 * num_train_images)

    return torch.utils.data.random_split(train_dataset, lengths=[train_size,
                                                           val_size])