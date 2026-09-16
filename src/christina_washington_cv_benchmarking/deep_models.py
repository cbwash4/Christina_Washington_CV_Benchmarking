
import torch.nn as nn
from torchvision import models


def get_model(model_name, num_classes=3, pretrained=True):

    weights = "DEFAULT" if pretrained else None
    name = model_name.lower()

    if name == "alexnet":
        model = models.alexnet(weights=weights)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )

    elif name == "vgg16":
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )

    elif name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(
            model.classifier.in_features, num_classes
        )

    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, num_classes
        )

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif name == "googlenet":
        model = models.googlenet(weights=weights, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if model.aux1 is not None:
            model.aux1.fc2 = nn.Linear(
                model.aux1.fc2.in_features, num_classes
            )

        if model.aux2 is not None:
            model.aux2.fc2 = nn.Linear(
                model.aux2.fc2.in_features, num_classes
            )

    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, num_classes
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
