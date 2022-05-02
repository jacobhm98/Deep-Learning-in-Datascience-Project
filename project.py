from cgi import test
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms, utils
# from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
# import tqdm

def download_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    print("Resnet downloaded")
    return model

def download_dataset():
    training_data = datasets.OxfordIIITPet(root = "data",split = "trainval",download =True)
    test_data = datasets.OxfordIIITPet(root = "data",split = "test", download =True)
    
    
    return training_data,test_data

def modify_model(model, n_classes):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=n_classes)
    return model

def train_model(model, train_dataloader, no_epoches, device):
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(no_epoches):
        model.train()
        for inputs, labels in train_dataloader:
                transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
                ])
                inputs = inputs.to(device).transforms.ToTensor()
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, inputs.to(device=device))
                print(loss)
    return model

def main():
    train_data, test_data = download_dataset()

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    print(train_dataloader)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    resnet_model = download_model()

    cat_dog_resnet_model = modify_model(resnet_model,37)
    model_ft = cat_dog_resnet_model.to(device)
    no_epoches = 200
    trained_model = train_model(cat_dog_resnet_model, train_dataloader, no_epoches, device)
    img, target = train_data[1]
    print(img)
    output = resnet_model(img)
    print(output)


if __name__ == '__main__':
    main()

