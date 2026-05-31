import torch.nn as nn


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        return self.encoder(features)
