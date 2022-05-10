from nn_lib import *
import data_utils


def main():
    # hyperparameters
    # set this to 2 if you want Cat dog classification else 37
    no_classes = 37  # should be 2 or 37 for binary vs multi class
    # classification
    batch_size = 64
    no_epochs = 2
    lr = 0.001
    loss_fxn = nn.CrossEntropyLoss()
    model_name = 'resnet18'
    cat_dog_dict = None

    # Set these according to your machine needs ( collab needs num_workers=2)
    num_workers = 2
    prefetch_factor = 2
    # download the train and test dataset and create batches for train and test dataset
    train_data, val_data, test_data = data_utils.download_dataset(
        augmentation=True,
        in_memory=False
    )

    # get the dictionary of cats and dogs to perform classification for cats and dogs comment id not needed
    if no_classes == 2:
        cat_dog_dict = data_utils.create_cat_dog_dict()

    # use GPU if available else use CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    # Download the pretrained model, where you can either freeze the
    # previous layers or fine tune the whole network,
    # by setting the freeze variable
    model, optimizer = download_model(model_name, 1, True, [0.001], no_classes)

    if no_classes == 2 or no_classes == 37:
        trained_model, train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_model(
            model,
            train_data, val_data,
            loss_fxn, optimizer,
            no_epochs, device,
            batch_size, cat_dog_dict,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor)
    else:
        raise ValueError("no_classes needs to be either 2 or 37")
    torch.save(trained_model, "models/trained_model.model")
    print("Model trained!!")


def investigate_effect_of_training_different_layers():
    # hyper parameters
    # set this to 2 if you want Cat dog classification else 37
    no_classes = 2  # should be 2 or 37 for binary vs multi class
    # classification
    batch_size = 64
    no_epochs = 2
    lr = 0.001
    loss_fxn = nn.CrossEntropyLoss()
    model_name = 'resnet18'
    cat_dog_dict = None

    # download the train and test dataset and create batches for train and test dataset
    train_data, test_data = data_utils.download_dataset(augmentation=True)
    train_data, val_data = data_utils.train_val_split(train_data)

    # use GPU if available else use CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    # Download the pretrained model, where you can either freeze the
    # previous layers or fine tune the whole network,
    # by setting the freeze variable
    for trained_layers in range(1, 6):
        model, optimizer = download_model(model_name, 1, True, [lr], no_classes)
        print_model_parameter_summary(model)

        trained_model, train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_model(model,
                                                                                              train_data, val_data,
                                                                                              loss_fxn, optimizer,
                                                                                              no_epochs, device,
                                                                                              batch_size,
                                                                                              cat_dog_dict)
        torch.save(trained_model, "models/trained_model.model")
        print("Model trained!!")

if __name__ == '__main__':
    main()
