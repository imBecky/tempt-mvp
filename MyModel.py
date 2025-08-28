import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_enc = nn.Sequential(
            nn.Linear(144, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        self.dec = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 32), nn.Sigmoid(),
            nn.Linear(32, 16), nn.Sigmoid(),
            nn.Linear(16, 144), nn.Sigmoid()
        )

    def forward(self, rgb):
        hidden_rgb = self.rgb_enc(rgb)
        j
