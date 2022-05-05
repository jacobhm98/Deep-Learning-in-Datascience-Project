import torch
from torch.utils.data import DataLoader, Dataset
from data_utils import gen_cat_dog_label
# from torchvision.transforms import ToTensor
import torch.nn as nn
from tqdm import tqdm
import copy




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
                running_loss = 0.0
                running_corrects = 0
                model.train()  # Set model to training mode
                for inputs, labels in train_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if cat_dog:
                        labels = gen_cat_dog_label(cat_dog_dict, labels)
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        preds = torch.argmax(outputs, 1, keepdim=True)
                        loss = loss_fxn(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds.T == labels.data)

                epoch_loss = running_loss / train_dataset_size
                epoch_acc = running_corrects.double() / train_dataset_size

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
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

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
