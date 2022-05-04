from doctest import Example
from pickletools import optimize
from pyexpat import model
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms, utils
# from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import os

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
        if(cat_dog == 1):
            cat_lis.append(id)
        else:
            dog_lis.append(id)
    a_file.close()
    cat_dog_dict = {"cat":cat_lis, "dog":dog_lis}
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
    training_data = datasets.OxfordIIITPet(root = "data",split = "trainval",download =True, transform = transform)
    test_data = datasets.OxfordIIITPet(root = "data",split = "test", download =True, transform = transform)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    example = iter(train_dataloader)
    sample, label = example.next()
    print(sample.shape, label.shape)
    return training_data,test_data, train_dataloader, test_dataloader

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

def train_model(model, train_dataloader, test_dataloader, loss_fxn, optimizer, no_epoches, device, batch_size, cat_dog_dict, cat_dog = True):
    '''
    Parameters:
    cat_dog_dict : {'cat':[cat_id_list], 'dog':[dog_id_list]}
    cat_dog : True if we want to classify into two classes.
    Return:
    Trained model
    '''
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for _ in tqdm(range(no_epoches)):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in train_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if cat_dog:
                        labels = gen_cat_dog_label(cat_dog_dict, labels)
                    dataset_size = len(train_dataloader)
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fxn(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / dataset_size
                    epoch_acc = running_corrects.double() / dataset_size

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
            else:
                model.eval()   # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in test_dataloader:
                    dataset_size = len(labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if cat_dog:
                        labels = gen_cat_dog_label(cat_dog_dict, labels)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    epoch_loss = running_loss / dataset_size
                    epoch_acc = running_corrects.double() / dataset_size

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # hyperparameters
    # set this to 2 if you want Cat dog classification else 37
    no_classes = 2
    batch_size = 64
    no_epoches = 10
    lr = 0.001
    loss_fxn = nn.CrossEntropyLoss()
    model_name = 'resnet18'

    # download the train and test dataset and create batches for train and test dataset
    train_data, test_data, train_dataloader, test_dataloader = download_dataset(batch_size)

    # get the dictionary of cats and dogs to perform classification for cats and dogs comment id not needed
    cat_dog_dict = create_cat_dog_dict()
    
    # use GPU if available else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download the pretrained model, where you can either freeze the 
    # previous layers or fine tune the whole network, 
    # by setting the freeze variable
    resnet_model = download_model(model_name, freeze = True, pretrained = True)
    # added a last layer to the network
    cat_dog_resnet_model = modify_model(resnet_model,no_classes)

    optimizer = optim.SGD(cat_dog_resnet_model.parameters(), lr = lr)
    trained_model = train_model(cat_dog_resnet_model, train_dataloader, test_dataloader, loss_fxn, optimizer, no_epoches, device, batch_size, cat_dog_dict)
    print("Model trained!!")


if __name__ == '__main__':
    main()

