import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.cross_attn = (
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        image_feat,
        text_feat
    ):

        image_feat = image_feat.unsqueeze(1)
        text_feat = text_feat.unsqueeze(1)

        attended, weights = (
            self.cross_attn(
                query=image_feat,
                key=text_feat,
                value=text_feat
            )
        )

        fused = self.norm(
            attended + image_feat
        )

        fused = fused.squeeze(1)

        return fused, weights