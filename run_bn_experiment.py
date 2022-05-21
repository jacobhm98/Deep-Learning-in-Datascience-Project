# Parameters
no_classes = 37  # should be 2 or 37 for binary vs multi class
device = "cuda"
# classification
batch_size = 64
no_epochs = 15

model_name = 'resnet34'

n_layers_to_train = 3
lr_per_layer = [1e-4]*3

# Performance parameters
num_workers = 6
prefetch_factor = 2
in_memory=False


# Imports
from data_utils import *
from nn_lib import *
from timm.data.transforms_factory import create_transform
import data_utils
from precise_bn import fix_model_bn_stats_on_dataset
from os import system

loss_fxn = nn.CrossEntropyLoss()

cat_dog_dict = None
import torch.nn as nn
tfs = transforms.Compose([
    transforms.RandomHorizontalFlip(),
])
# download the train and test dataset and create batches for train and test dataset
train_data, val_data, test_data = download_dataset(
  augmentation=False, in_memory=in_memory,train_transforms=None, num_train_examples=80
)

# get the dictionary of cats and dogs to perform classification for cats and dogs comment id not needed
if no_classes == 2:
    cat_dog_dict = data_utils.create_cat_dog_dict()

# use GPU if available else use CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set these according to your machine needs ( collab needs num_workers=2)

# If you want to disable parallelization in the dataloader all together
# set num_workers = 0 and prefetch_factor=2

# Download the pretrained model, where you can either freeze the
# previous layers or fine tune the whole network,
# by setting the freeze variable

def train():
    global train_data, val_data, loss_fxn, no_epochs, \
        device, batch_size, cat_dog_dict, num_workers, prefetch_factor

    model, optimizer = download_model(model_name, n_layers_to_train, True,
                                      lr_per_layer, no_classes,
                                      fine_tune_batch_norm=fine_tune_batch_norm)

    trained_model, _, _, _, _ = train_model(
        model,
        train_data, val_data,
        loss_fxn, optimizer,
        no_epochs, device,
        batch_size, cat_dog_dict,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    return trained_model


fine_tune_batch_norm = True
print("training with batch norm? ", fine_tune_batch_norm)
model = train()
os.system("mkdir results/with_bn")
os.system("cp val_metrics.csv results/with_bn/val_metrics.csv")
os.system("cp train_metrics.csv results/with_bn/train_metrics.csv")

torch.save(model, f"models/exp4_bn{fine_tune_batch_norm}")

fine_tune_batch_norm = False
print("training with batch norm? ", fine_tune_batch_norm)
model = train()
os.system("mkdir results/without_bn")
os.system("cp val_metrics.csv results/without_bn/val_metrics.csv")
os.system("cp train_metrics.csv results/without_bn/train_metrics.csv")

torch.save(model, f"models/exp4_bn{fine_tune_batch_norm}")

print("Restoring model to the one with BN")
model = torch.load("models/exp4_bnTrue")



print("=============Accuracies before fixing bn ==========")
print("Train accuracy")
print(test_loss_and_accuracy(train_data, model, loss_fxn, device, cat_dog_dict, num_workers=num_workers, prefetch_factor=prefetch_factor))
print("Val accuracy")
print(test_loss_and_accuracy(val_data, model, loss_fxn, device, cat_dog_dict, num_workers=num_workers, prefetch_factor=prefetch_factor))
print("Test accuracy")
print(test_loss_and_accuracy(test_data, model, loss_fxn, device, cat_dog_dict, num_workers=num_workers, prefetch_factor=prefetch_factor))


print("=========Fixing model stats on Val==========")
fix_model_bn_stats_on_dataset(model, val_data, batch_size, device=device)

print("Train accuracy")
print(test_loss_and_accuracy(train_data, model, loss_fxn, device, cat_dog_dict,
                             num_workers=num_workers, prefetch_factor=prefetch_factor))
print("Val accuracy")
print(test_loss_and_accuracy(val_data, model, loss_fxn, device, cat_dog_dict, num_workers=num_workers, prefetch_factor=prefetch_factor))

print("=========Fixing model stats on Train==========")
fix_model_bn_stats_on_dataset(model, train_data, batch_size, device=device)

print("Train accuracy")
print(test_loss_and_accuracy(train_data, model, loss_fxn, device, cat_dog_dict,
                             num_workers=num_workers, prefetch_factor=prefetch_factor))
print("Val accuracy")
print(test_loss_and_accuracy(val_data, model, loss_fxn, device, cat_dog_dict, num_workers=num_workers, prefetch_factor=prefetch_factor))

## Plotting results

import pandas as pd
def ldcsv(fname):
  df = pd.read_csv(fname)
  return df.drop(columns=df.columns[0])

train_df_bn = ldcsv("results/with_bn/train_metrics.csv")
train_df_nobn = ldcsv("results/without_bn/train_metrics.csv")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(train_df_bn, label="With BN")
plt.plot(train_df_nobn, label="Without BN")
plt.legend(loc="upper right")
plt.savefig("figures/bn_train_stats.pdf")

val_df_bn = ldcsv("results/with_bn/val_metrics.csv")
val_df_no_bn = ldcsv("results/without_bn/val_metrics.csv")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(val_df_bn['val loss'], label="With BN")
plt.plot(val_df_no_bn['val loss'], label="Without BN")
plt.legend(loc="upper right")
plt.savefig("figures/bn_val_loss_stats.pdf")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(val_df_bn['val_acc'], label="With BN")
plt.plot(val_df_no_bn['val_acc'], label="Without BN")
plt.legend(loc="upper right")
plt.savefig("figures/bn_val_acc_stats.pdf")