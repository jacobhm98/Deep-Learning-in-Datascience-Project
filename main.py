from torchvision import transforms

from data_utils import print_dataset_summary
from nn_lib import *
import data_utils

def train_aug(aug, n_classes, filename):
    # hyperparameters
    # set this to 2 if you want Cat dog classification else 37
    no_classes = n_classes  # should be 2 or 37 for binary vs multi class
    # classification
    batch_size = 64
    no_epochs = 2
    lr = 0.001
    loss_fxn = nn.CrossEntropyLoss()
    model_name = 'resnet18'
    cat_dog_dict = None

    # download the train and test dataset and create batches for train and test dataset
    train_data, val_data, test_data = data_utils.download_dataset(augmentation = True)

    # get the dictionary of cats and dogs to perform classification for cats and dogs comment id not needed
    if no_classes == 2:
        cat_dog_dict = data_utils.create_cat_dog_dict()

    # use GPU if available else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    # Download the pretrained model, where you can either freeze the
    # previous layers or fine tune the whole network,
    # by setting the freeze variable
    n_layers_to_train = 3
    model, optimizer = download_model(model_name, 3, True, [0.001] *n_layers_to_train,
                                      no_classes)

    if no_classes == 2:
        trained_model, train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_model(model,
                                                                                              train_data, val_data,
                                                                                              loss_fxn, optimizer,
                                                                                              no_epochs, device,
                                                                                              batch_size, cat_dog_dict,
                                                                                              train_metrics_filename=str(filename)+"_train.csv",
                val_metrics_filename=str(filename)+"_val.csv")
    elif no_classes == 37:
        trained_model, train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_model(model,
                                                                                              train_data, val_data,
                                                                                              loss_fxn, optimizer,
                                                                                              no_epochs, device,
                                                                                              batch_size,
                                                                                              cat_dog_dict,train_metrics_filename=str(filename)+"_train.csv",
                val_metrics_filename=str(filename)+"_val.csv")
    else:
        raise ValueError("no_classes needs to be either 2 or 37")
    torch.save(trained_model, "models/trained_model.model")
    print("Model trained!!")

def investigate_data_augumentation_effect():
    no_classes = 37
    batch_size = 64
    no_epochs = 2
    lr = 0.001
    #  for 37 classes
    # with data augumentation 
    file = str(37)+"_aug"
    train_aug(True, no_classes, file)
    # with different data augumentation -> check normalisation transform
    # without data augumentation
    file = str(37)+"_no_aug"
    train_aug(False, no_classes, file)
    # for 2 classes
    # with data augumentation 
    no_classes = 2
    file = str(2)+"_aug"
    train_aug(True, no_classes, file)
    # with different data augumentation -> check normalisation transform
    # without data augumentation
    file = str(2)+"_no_aug"
    train_aug(False, no_classes, file)


def main():
    # data_utils.create_val_plot("results/val_metrics_5.csv")
    # hyperparameters
    # set this to 2 if you want Cat dog classification else 37
    no_classes = 37  # should be 2 or 37 for binary vs multi class
    # classification
    batch_size = 64
    no_epochs = 10
    loss_fxn = nn.CrossEntropyLoss()
    model_name = 'resnet18'
    cat_dog_dict = None

    # Set these according to your machine needs ( collab needs num_workers=2)
    num_workers = 6
    prefetch_factor = 2

    tfs = transforms.Compose([
        #transforms.Resize((255,255)),
        #transforms.ColorJitter(),
        #transforms.GaussianBlur(kernel_size=[5,5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(),
        transforms.RandomRotation(45),
        #transforms.RandomResizedCrop((255,255), scale=(0.7,0.7)),
    ])

    # download the train and test dataset and create batches for train and test dataset
    train_data, val_data, test_data = data_utils.download_dataset(
        augmentation=True,
        in_memory=False,
        train_transforms=tfs
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
    model, optimizer = download_model(model_name, 2, True, [1e-4]*2,
                                      no_classes, fine_tune_batch_norm=True)

    if no_classes == 2 or no_classes == 37:
        trained_model, train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_model_pseudolabelling(
            model,
            train_data, val_data,
            loss_fxn, optimizer,
            no_epochs, device,
            batch_size, tfs, cat_dog_dict,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor)
    else:
        raise ValueError("no_classes needs to be either 2 or 37")
    torch.save(trained_model, "models/trained_model.model")
    print("Model trained!!")


def investigate_effect_of_training_different_layers():
    # hyper parameters
    # set this to 2 if you want Cat dog classification else 37
    no_classes = 37  # should be 2 or 37 for binary vs multi class
    # classification
    batch_size = 64
    no_epochs = 10
    lr = 0.001
    loss_fxn = nn.CrossEntropyLoss()
    model_name = 'resnet34'
    cat_dog_dict = None

    # download the train and test dataset and create batches for train and test dataset
    train_data, val_data, test_data = data_utils.download_dataset(augmentation=False, unlabelled_percent=20)


    # use GPU if available else use CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # Download the pretrained model, where you can either freeze the
    # previous layers or fine tune the whole network,
    # by setting the freeze variable
    for trained_layers in range(1, 6):
        model, optimizer = download_model(model_name, 1, True, [lr], no_classes)
        print_model_parameter_summary(model)

        # trained_model, train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_model(model,
        #                                                                                       train_data, val_data,
        #                                                                                       loss_fxn, optimizer,
        #                                                                                       no_epochs, device,
        #                                                                                       batch_size,
        #                                                                                       cat_dog_dict, val_metrics_filename="val_metrics_" + str(trained_layers) + ".csv", train_metrics_filename="train_metrics_" + str(trained_layers) + ".csv")

        trained_model, train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_model_pseudolabelling(model,
                                                                                              train_data, val_data,
                                                                                              loss_fxn, optimizer,
                                                                                              no_epochs, device,
                                                                                              batch_size,
                                                                                              cat_dog_dict, val_metrics_filename="val_metrics_" + str(trained_layers) + ".csv", train_metrics_filename="train_metrics_" + str(trained_layers) + ".csv")        
        torch.save(trained_model, "models/trained_model.model")
        print("Model trained!!")

if __name__ == '__main__':
    main()
