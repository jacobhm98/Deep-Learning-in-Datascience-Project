from nn_lib import download_model, print_model_parameter_summary


def main():
    model_name = "resnet152"
    no_classes = 37
    model, optimizer = download_model(model_name, 6, True, [1e-4] * 6,
                                      no_classes, fine_tune_batch_norm=False)
    print_model_parameter_summary(model)

if __name__ == '__main__':
    main()