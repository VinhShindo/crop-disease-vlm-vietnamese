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

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Quan trọng: batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, text_feat):
        """
        Args:
            image_feat: [batch_size, embed_dim] or [batch_size, 1, embed_dim]
            text_feat: [batch_size, embed_dim] or [batch_size, 1, embed_dim]
        Returns:
            fused: [batch_size, embed_dim]
            weights: attention weights
        """
        # Đảm bảo input là 3D [batch_size, seq_len, embed_dim]
        if image_feat.dim() == 2:
            image_feat = image_feat.unsqueeze(1)  # [B, 1, D]
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)    # [B, 1, D]
        
        # Cross attention: query từ image, key/value từ text
        attended, weights = self.cross_attn(
            query=image_feat,      # [B, 1, D]
            key=text_feat,         # [B, 1, D]
            value=text_feat        # [B, 1, D]
        )
        
        # Residual connection + layer norm
        fused = self.norm(image_feat + self.dropout(attended))
        
        # Squeeze back to 2D
        fused = fused.squeeze(1)   # [B, D]
        
        return fused, weights