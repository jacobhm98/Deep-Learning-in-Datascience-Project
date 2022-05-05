import torch
from torch.utils.data import DataLoader, Dataset
from data_utils import gen_cat_dog_label
# from torchvision.transforms import ToTensor
import torch.nn as nn
from tqdm import tqdm
import copy


# note that in the data/annotations/list.txt has filename | 1:37 Class ids | 1:Cat 2:Dog | 1-25:Cat 1:12:Dog
# so function given an id get whether cat or dog, 
# creating a dictionary with {'cat':[list of ids], 'dog':[list of ids]}, computed in the beginning first and then retrive later.
def create_cat_dog_dict():
    '''
    Returns:
    {'cat':[list of ids], 'dog':[list of ids]}
    '''
    cat_lis = []
    dog_lis = []
    list_file_path = "data/oxford-iiit-pet/annotations/list.txt"
    a_file = open(list_file_path)
    lines = a_file.readlines()[6:]
    for line in lines:
        id = line.rstrip().split()[1]
        if id in cat_lis or id in dog_lis:
            continue
        cat_dog = line.rstrip().split()[2]
        if (cat_dog == '1'):
            cat_lis.append(id)
        else:
            dog_lis.append(id)
    a_file.close()
    cat_dog_dict = {"cat": cat_lis, "dog": dog_lis}
    return cat_dog_dict


def download_model(model_name, freeze, pretrained):
    '''
    Parameters:
    model_name : string -  name of the model
    freeze : Boolean value- should we freeze all the other layers in the model to prevent them from getting trained
    pretrained : Boolean value- should we use a pretrained model or not

    Returns:
    Resnet-18 model
    '''
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=pretrained)
    print("Resnet downloaded")
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        print("All the layers won't be trained, except the last layer")
    return model


def modify_model(model, n_classes):
    '''
    Parameters:
    model: resnet_model
    n_classes: number of labels for the dataset/ number of nodes in the last layer of the NN.

    Returns:
    model with final layer added.
    '''
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=n_classes)
    return model


def train_model(model, train_data, val_data, loss_fxn, optimizer, no_epochs, device, batch_size, cat_dog_dict,
                cat_dog=True):
    '''
    Parameters:
    cat_dog_dict : {'cat':[cat_id_list], 'dog':[dog_id_list]}
    cat_dog : True if we want to classify into two classes.
    Return:
    Trained model
    '''
    train_dataset_size = len(train_data)
    val_dataset_size = len(val_data)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for _ in tqdm(range(no_epochs)):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                for inputs, labels in train_dataloader:
                    running_loss = 0.0
                    running_corrects = 0
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if cat_dog:
                        labels = gen_cat_dog_label(cat_dog_dict, labels)
                    train_dataset_size = len(train_dataloader)
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fxn(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / train_dataset_size
                epoch_acc = running_corrects.double() / train_dataset_size

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            else:
                model.eval()  # Set model to evaluate mode

                # Iterate over data.
                for inputs, labels in test_dataloader:
                    running_loss = 0.0
                    running_corrects = 0
                    train_dataset_size = len(labels)
                    inputs = inputs.to(device)
                    if cat_dog:
                        labels = gen_cat_dog_label(cat_dog_dict, labels)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / val_dataset_size
                epoch_acc = running_corrects.double() / val_dataset_size

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
