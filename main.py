from nn_lib import *
from torch import optim
import data_utils


def main():
    # hyperparameters
    # set this to 2 if you want Cat dog classification else 37
    no_classes = 2
    batch_size = 64
    no_epochs = 10
    lr = 0.001
    loss_fxn = nn.CrossEntropyLoss()
    model_name = 'resnet18'

    # download the train and test dataset and create batches for train and test dataset
    train_data, test_data = data_utils.download_dataset()
    train_data, val_data = data_utils.train_val_split(train_data)

    # get the dictionary of cats and dogs to perform classification for cats and dogs comment id not needed
    cat_dog_dict = data_utils.create_cat_dog_dict()

    # use GPU if available else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download the pretrained model, where you can either freeze the
    # previous layers or fine tune the whole network,
    # by setting the freeze variable
    resnet_model = download_model(model_name, freeze=True, pretrained=True)
    # added a last layer to the network
    cat_dog_resnet_model = modify_model(resnet_model, no_classes)

    optimizer = optim.SGD(cat_dog_resnet_model.parameters(), lr=lr)
    trained_model = train_model(cat_dog_resnet_model, train_data, val_data, loss_fxn, optimizer, no_epochs, device,
                                batch_size, cat_dog_dict)
    #TODO this save statemnt doesnt currently work
    torch.save(trained_model, "models/trained_model.model")
    print("Model trained!!")


if __name__ == '__main__':
    main()
