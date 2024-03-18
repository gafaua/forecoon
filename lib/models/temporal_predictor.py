import torch
import torch.nn as nn

from lib.models.feature_extractors import get_feature_extractor


class TemporalPredictor(nn.Module):
    def __init__(self,
                 backbone,
                 dim,
                 hidden_dim,) -> None:
        super().__init__()
        self.backbone = get_feature_extractor(backbone)
        self.prediction = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, img1, img2):
        z1 = self.backbone(img1)
        z2 = self.backbone(img2)#.detach()
        preds = self.prediction(z2-z1)
        return preds