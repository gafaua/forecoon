import torch
from lib.models.networks.simple_cnn import SimpleCNN
from lib.models.networks.vision_transformer import vit_base, vit_small, vit_tiny
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from torchvision.models.vgg import vgg11_bn, VGG11_BN_Weights
import ssl

from lib.models.siamese_ema import SiameseEMA

ssl._create_default_https_context = ssl._create_unverified_context


def _load_checkpoint(model, path):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_dict"])

    print("="*100)
    print(f"Loading model from checkpoint {path}")
    print("="*100)

    return model


def _wrap_model_1to3channels(model):
    return nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=1, bias=False),
        model
    )

def get_resnet18():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    return _wrap_model_1to3channels(model)

def get_moco_encoder(backbone: str, weights: str):
    model = SiameseEMA(
            base_encoder=get_feature_extractor(backbone),
    )
    model = _load_checkpoint(model, weights)
    model = model.encoder_q
    model.eval()

    return model


def get_resnet34():
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Identity()

    return _wrap_model_1to3channels(model)

def get_vgg11():
    model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
    #print(model.classifier[-1])
    del model.classifier[-1]
    return _wrap_model_1to3channels(model)

def get_simple():
    model = SimpleCNN()
    return model

_feature_extractors = dict(
    vit_tiny=vit_tiny,
    vit_small=vit_small,
    vit_base=vit_base,
    resnet18=get_resnet18,
    resnet34=get_resnet34,
    vgg11=get_vgg11,
    simple=get_simple,
)


def get_feature_extractor(name):
    return _feature_extractors[name]()
