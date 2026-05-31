import torch.nn as nn


class TextProjection(nn.Module):

    def __init__(
        self,
        input_dim=768,
        output_dim=768,
        dropout=0.2
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(
                input_dim,
                output_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.proj(x)