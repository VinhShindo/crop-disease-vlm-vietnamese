import timm
import torch
import torch.nn as nn


class EfficientNetEncoder(nn.Module):
    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        return_feature_maps=True  # Thêm flag này
    ):
        super().__init__()
        self.return_feature_maps = return_feature_maps
        
        # Load model without global pooling
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""  # Không global pool để lấy feature maps
        )
        
        # Lấy feature dimension từ backbone
        self.output_dim = self.backbone.num_features
        
        # Thêm adaptive pooling để đưa về kích thước cố định
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # [B, C, 7, 7]

    def forward(self, images):
        # Get feature maps [B, C, H, W]
        features = self.backbone(images)
        
        if self.return_feature_maps:
            # Return feature maps for spatial attention
            return features  # [B, C, H, W]
        else:
            # Global pooling
            pooled = features.mean(dim=[2, 3])  # [B, C]
            return pooled