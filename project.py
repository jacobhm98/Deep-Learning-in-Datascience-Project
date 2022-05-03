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

def train_model(model, train_dataloader, test_dataloader, loss_fxn, optimizer, no_epoches, device, batch_size):
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
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fxn(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / batch_size
                    epoch_acc = running_corrects.double() / batch_size

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
            else:
                model.eval()   # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    epoch_loss = running_loss / batch_size
                    epoch_acc = running_corrects.double() / batch_size

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
    batch_size = 64
    no_epoches = 10
    lr = 0.001
    loss_fxn = nn.CrossEntropyLoss()
    model_name = 'resnet18'
    # download the train and test dataset and create batches for train and test dataset
    train_data, test_data, train_dataloader, test_dataloader = download_dataset(batch_size)
    
    # use GPU if available else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download the pretrained model, where you can either freeze the 
    # previous layers or fine tune the whole network, 
    # by setting the freeze variable
    resnet_model = download_model(model_name, freeze = True, pretrained = True)
    # added a last layer to the network
    cat_dog_resnet_model = modify_model(resnet_model,37)

    optimizer = optim.SGD(cat_dog_resnet_model.parameters(), lr = lr)
    trained_model = train_model(cat_dog_resnet_model, train_dataloader, test_dataloader, loss_fxn, optimizer, no_epoches, device, batch_size)
    print("Model trained!!")


if __name__ == '__main__':
    main()

