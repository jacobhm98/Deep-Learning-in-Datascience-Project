import imp
import torch
from torch.utils.data import DataLoader, Dataset
from data_utils import gen_cat_dog_label
# from torchvision.transforms import ToTensor
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm
import copy
import pandas as pd


def is_layer_frozen(layer: nn.Module):
    return all([x.requires_grad == True for x in layer.parameters()])


def download_model(model_name, n_layers_to_train, pretrained, lr_per_layer,
                   n_classes, fine_tune_batch_norm=False):
    '''
    Parameters:
    model_name : string -  name of the model
    pretrained : Boolean value- should we use a pretrained model or not
    n_layers_to_train : number of trainable layers
    n_classes :
    fine_tune_batch_norm: True if the batch norm layer parameters are to be
    tuned
    Returns:
    Resnet-18 model
    '''
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=pretrained)
    print("Resnet downloaded")
    parameter_groups = []

    # Freeze the first layers
    model.conv1.weight.requires_grad = False
    model.bn1.weight.requires_grad = False
    model.bn1.bias.requires_grad = False

    assert n_layers_to_train >= 0 and n_layers_to_train < 4
    n_layers_to_freeze = 4 - n_layers_to_train + 1
    for i in range(n_layers_to_freeze):
        layer = model.__getattr__(f"layer{i + 1}")
        for param in layer.parameters():
            param.requires_grad = False

    for i in range(n_layers_to_train - 1):
        layer = model.__getattr__(f"layer{4 - i}")
        parameter_groups.append(
            {'params': layer.parameters(), 'lr': lr_per_layer[i]}
        )

    # Dealing with batch norm parameters
    if not fine_tune_batch_norm:
        for name, param in model.named_parameters():
            if "bn" in name:
                param.requires_grad = False
    print("Frozen ", 5 - n_layers_to_train, "layers")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=n_classes)
    # Add final layer and input layer
    parameter_groups.append(
        {'params': model.fc.parameters(), 'lr': lr_per_layer[-1]}
    )

    optimizer = Adam(parameter_groups)
    return model, optimizer


def print_model_parameter_summary(model):
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)



def train_model(model, train_data, val_data, loss_fxn, optimizer, no_epochs, device, batch_size, cat_dog_dict,
                cat_dog=True, train_metrics_filename="train_metrics.csv",
                val_metrics_filename="val_metrics.csv", num_workers=2,
                prefetch_factor=2):
    '''
    Parameters:
    cat_dog_dict : {'cat':[cat_id_list], 'dog':[dog_id_list]}
    Return:
    Trained model, loss_arr, acc_arr
    '''

    # If there is a cat dog dict then we want to do classification on 2 species
    cat_dog = cat_dog_dict is not None

    model.to(device)
    train_dataset_size = len(train_data)
    val_dataset_size = len(val_data)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  prefetch_factor=prefetch_factor)

    test_dataloader = DataLoader(val_data, batch_size=batch_size,
                                 num_workers=8, prefetch_factor=2)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_loss_arr = []
    val_loss_arr = []
    train_acc_arr = []
    val_acc_arr = []

    progress_bar = tqdm(range(no_epochs))
    for _ in progress_bar:
        for phase in ['train', 'val']:
            if phase == 'train':
                running_loss = 0.0
                running_corrects = 0
                model.train()  # Set model to training mode
                for inputs, labels in tqdm(train_dataloader):
                    inputs = inputs.to(device)
                    if cat_dog:
                        labels = gen_cat_dog_label(cat_dog_dict, labels)

                    labels = labels.to(device)
                    # with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, 1, keepdim=True)
                    loss = loss_fxn(outputs, labels)

                    # Keep track of loss metric
                    train_loss_arr.append(loss.item())

                    # Do backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    progress_bar.set_postfix_str(loss.item())
                    running_corrects += torch.sum(preds.T == labels.data)

                # Accuracy loss
                epoch_loss = running_loss / train_dataset_size
                # epoch_acc = running_corrects.double() / train_dataset_size
                # train_acc_arr.append(epoch_acc)

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
            else:
                model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    if cat_dog:
                        labels = gen_cat_dog_label(cat_dog_dict, labels)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1, keepdim=True)
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds.T == labels.data)


                epoch_loss = running_loss / val_dataset_size
                epoch_acc = running_corrects.double() / val_dataset_size

                val_acc_arr.append(epoch_acc.item())
                val_loss_arr.append(epoch_loss)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())


    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    df_train = pd.DataFrame({
        'train_loss': train_loss_arr,
    })

    df_val = pd.DataFrame({
        'val_acc': val_acc_arr,
        'val loss': val_loss_arr
    })
    df_train.to_csv(train_metrics_filename)
    df_val.to_csv(val_metrics_filename)
    return model, train_acc_arr,train_loss_arr, val_acc_arr, val_loss_arr
