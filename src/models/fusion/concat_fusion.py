import torch
import torch.nn as nn


class ConcatFusion(nn.Module):

    def __init__(
        self,
        image_dim=768,
        text_dim=768,
        output_dim=768
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(
                image_dim + text_dim,
                output_dim
            ),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(
        self,
        image_feat,
        text_feat
    ):

        fused = torch.cat(
            [
                image_feat,
                text_feat
            ],
            dim=1
        )

        return self.fusion(fused)