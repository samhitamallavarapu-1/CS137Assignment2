import torch.nn as nn
import torchvision.models as tv_models


def get_model(model_name: str, num_classes: int, pretrained: bool):
    model_name = model_name.lower()

    if model_name == "densenet121":
        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = tv_models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet152":
        weights = tv_models.ResNet152_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet152(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def set_last_layer_only(model):
    freeze_all(model)

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError("Could not find final classifier layer.")


def set_full_finetune(model):
    unfreeze_all(model)


def get_unfreeze_blocks(model):
    if hasattr(model, "features") and hasattr(model, "classifier"):  # DenseNet
        blocks = [model.classifier]
        if hasattr(model.features, "denseblock4"):
            blocks += [
                model.features.denseblock4,
                model.features.transition3,
                model.features.denseblock3,
                model.features.transition2,
                model.features.denseblock2,
                model.features.transition1,
                model.features.denseblock1,
                model.features.conv0,
            ]
        return blocks

    if hasattr(model, "fc") and hasattr(model, "layer4"):  # ResNet
        return [
            model.fc,
            model.layer4,
            model.layer3,
            model.layer2,
            model.layer1,
            nn.Sequential(model.conv1, model.bn1),
        ]

    raise ValueError("Unsupported architecture for gradual unfreezing.")


def set_gradual_unfreeze(model, epoch: int, unfreeze_every: int = 10):
    freeze_all(model)
    blocks = get_unfreeze_blocks(model)

    n_unfrozen = min(len(blocks), 1 + (epoch - 1) // unfreeze_every)

    for block in blocks[:n_unfrozen]:
        for p in block.parameters():
            p.requires_grad = True

    return n_unfrozen