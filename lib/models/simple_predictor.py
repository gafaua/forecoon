import torch.nn as nn

from lib.models.feature_extractors import get_feature_extractor


class SimplePredictor(nn.Module):
    def __init__(self,
                 backbone,
                 dim,
                 hidden_dim,
                 output_dim,) -> None:
        super().__init__()
        self.backbone = get_feature_extractor(backbone)
        self.head = nn.Sequential(
            nn.Linear(dim, output_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.head(self.backbone(x))