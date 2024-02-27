from lib.models.vision_transformer import vit_base, vit_small, vit_tiny
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_resnet18():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    #model.conv1 = nn.Conv2d(1, , kernel_size=7, stride=2, padding=3, bias=False)
    model = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=1, bias=False),
        model
    )
    return model


_feature_extractors = dict(
    vit_tiny=vit_tiny,
    vit_small=vit_small,
    vit_base=vit_base,
    resnet18=get_resnet18,
)


def get_feature_extractor(name):
    return _feature_extractors[name]()
