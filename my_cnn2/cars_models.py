# import torch.nn as nn
# import torchvision.models as tv_models


# def get_model(model_name: str, num_classes: int, pretrained: bool):
#     model_name = model_name.lower()

#     if model_name == "densenet121":
#         weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
#         model = tv_models.densenet121(weights=weights)
#         in_features = model.classifier.in_features
#         model.classifier = nn.Linear(in_features, num_classes)
#         return model

#     if model_name == "resnet152":
#         weights = tv_models.ResNet152_Weights.DEFAULT if pretrained else None
#         model = tv_models.resnet152(weights=weights)
#         in_features = model.fc.in_features
#         model.fc = nn.Linear(in_features, num_classes)
#         return model

#     raise ValueError(f"Unsupported model_name: {model_name}")


# def freeze_all(model):
#     for p in model.parameters():
#         p.requires_grad = False


# def unfreeze_all(model):
#     for p in model.parameters():
#         p.requires_grad = True


# def set_last_layer_only(model):
#     freeze_all(model)

#     if hasattr(model, "classifier"):
#         for p in model.classifier.parameters():
#             p.requires_grad = True
#     elif hasattr(model, "fc"):
#         for p in model.fc.parameters():
#             p.requires_grad = True
#     else:
#         raise ValueError("Could not find final classifier layer.")


# def set_full_finetune(model):
#     unfreeze_all(model)


# def get_unfreeze_blocks(model):
#     if hasattr(model, "features") and hasattr(model, "classifier"):  # DenseNet
#         blocks = [model.classifier]
#         if hasattr(model.features, "denseblock4"):
#             blocks += [
#                 model.features.denseblock4,
#                 model.features.transition3,
#                 model.features.denseblock3,
#                 model.features.transition2,
#                 model.features.denseblock2,
#                 model.features.transition1,
#                 model.features.denseblock1,
#                 model.features.conv0,
#             ]
#         return blocks

#     if hasattr(model, "fc") and hasattr(model, "layer4"):  # ResNet
#         return [
#             model.fc,
#             model.layer4,
#             model.layer3,
#             model.layer2,
#             model.layer1,
#             nn.Sequential(model.conv1, model.bn1),
#         ]

#     raise ValueError("Unsupported architecture for gradual unfreezing.")


# def set_gradual_unfreeze(model, epoch: int, unfreeze_every: int = 10):
#     freeze_all(model)
#     blocks = get_unfreeze_blocks(model)

#     n_unfrozen = min(len(blocks), 1 + (epoch - 1) // unfreeze_every)

#     for block in blocks[:n_unfrozen]:
#         for p in block.parameters():
#             p.requires_grad = True

#     return n_unfrozen

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


def get_head_module(model):
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "fc"):
        return model.fc
    raise ValueError("Could not find model head.")


def get_backbone_blocks_output_to_input(model):
    """
    Return backbone blocks ordered from output-side toward input-side.
    Head is NOT included here.
    """
    # DenseNet121
    if hasattr(model, "features") and hasattr(model, "classifier"):
        return [
            model.features.denseblock4,
            model.features.transition3,
            model.features.denseblock3,
            model.features.transition2,
            model.features.denseblock2,
            model.features.transition1,
            model.features.denseblock1,
            model.features.conv0,
        ]

    # ResNet152
    if hasattr(model, "layer4") and hasattr(model, "fc"):
        return [
            model.layer4,
            model.layer3,
            model.layer2,
            model.layer1,
            nn.Sequential(model.conv1, model.bn1),
        ]

    raise ValueError("Unsupported architecture for block extraction.")


def set_gradual_unfreeze(model, epoch: int, unfreeze_every: int = 10):
    """
    Existing cumulative gradual unfreezing:
    head always trainable, then add more output-side blocks over time.
    """
    freeze_all(model)

    head = get_head_module(model)
    for p in head.parameters():
        p.requires_grad = True

    blocks = get_backbone_blocks_output_to_input(model)
    n_unfrozen = min(len(blocks), 1 + (epoch - 1) // unfreeze_every)

    for block in blocks[:n_unfrozen]:
        for p in block.parameters():
            p.requires_grad = True

    return {
        "mode": "gradual",
        "n_backbone_blocks_trainable": n_unfrozen,
        "trainable_block_indices": list(range(n_unfrozen)),
    }


def set_sliding_window_finetune(
    model,
    epoch: int,
    slide_every: int = 10,
    window_size: int = 2,
):
    """
    Head always trains.
    A fixed-size window of backbone blocks trains and slides from output-side
    toward input-side over time.

    Example:
      blocks = [b0, b1, b2, b3, b4]
      window_size=2, slide_every=10

      epochs  1-10  -> head + [b0, b1]
      epochs 11-20  -> head + [b1, b2]
      epochs 21-30  -> head + [b2, b3]
      epochs 31-40  -> head + [b3, b4]
      epochs 41+    -> head + [b3, b4]
    """
    freeze_all(model)

    head = get_head_module(model)
    for p in head.parameters():
        p.requires_grad = True

    blocks = get_backbone_blocks_output_to_input(model)
    n_blocks = len(blocks)

    if n_blocks == 0:
        return {
            "mode": "sliding_window",
            "window_start": None,
            "window_end": None,
            "trainable_block_indices": [],
        }

    window_size = max(1, min(window_size, n_blocks))
    step_idx = max(0, (epoch - 1) // slide_every)

    max_start = max(0, n_blocks - window_size)
    start = min(step_idx, max_start)
    end = start + window_size

    for block in blocks[start:end]:
        for p in block.parameters():
            p.requires_grad = True

    return {
        "mode": "sliding_window",
        "window_start": start,
        "window_end": end - 1,
        "trainable_block_indices": list(range(start, end)),
    }