import torch


def apply_ckpt(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt)
    # print(f'모델을 성공적으로 불러왔습니다.')
    return model


def apply_device(model, device):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 10:
            print("Multi-Device")
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)
    else:
        model = model.to(device)
    return model


def efficientnet(model_name="efficientnet-b4", mode="train"):
    from efficientnet_pytorch import EfficientNet
    """
    efficientnet-b0 ~ b7
    """
    if mode == "train":
        model = EfficientNet.from_pretrained(model_name, num_classes=4)
    else:
        model = EfficientNet.from_name(model_name, num_classes=4)

    return model


def unet_load(device, mode="train"):
    import segmentation_models_pytorch as smp

    aux_params = dict(
        # pooling='avg',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        activation='softmax',  # activation function, default is None
        # classes=4,  # define number of output labels
    )

    if mode == 'train':
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
            activation='sigmoid',
        )

    else:
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
            activation='sigmoid',
        )
    model = apply_device(model, device)
    return model


if __name__ == '__main__':
    pass
