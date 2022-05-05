import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from project import create_cat_dog_dict, get_dataloaders, download_model, \
    modify_model, train_model


class MyTestCase(unittest.TestCase):
    def test_training_happens_without_errors(self):
        # hyperparameters
        # set this to 2 if you want Cat dog classification else 37
        no_classes = 2
        batch_size = 64
        no_epoches = 1
        lr = 0.001
        loss_fxn = nn.CrossEntropyLoss()
        model_name = 'resnet18'

        # download the train and test dataset and create batches for train and test dataset
        train_data, test_data, train_dataloader, test_dataloader = get_dataloaders(
            batch_size)

        # get the dictionary of cats and dogs to perform classification for cats and dogs comment id not needed
        cat_dog_dict = create_cat_dog_dict()

        # use GPU if available else use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download the pretrained model, where you can either freeze the
        # previous layers or fine tune the whole network,
        # by setting the freeze variable
        resnet_model = download_model(model_name, freeze=True, pretrained=True)
        # added a last layer to the network
        cat_dog_resnet_model = modify_model(resnet_model, no_classes)

        optimizer = optim.SGD(cat_dog_resnet_model.parameters(), lr=lr)
        trained_model = train_model(cat_dog_resnet_model, train_dataloader,
                                    test_dataloader, loss_fxn, optimizer,
                                    no_epoches, device, batch_size,
                                    cat_dog_dict)
        print("Model trained!!")


if __name__ == '__main__':
    unittest.main()
