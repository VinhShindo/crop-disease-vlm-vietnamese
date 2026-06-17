import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision.efficientnet_encoder import EfficientNetEncoder
from models.vision.vision_projection import VisionProjection
from models.text.phobert_encoder import PhoBERTEncoder
from models.text.text_projection import TextProjection
from models.fusion.concat_fusion import ConcatFusion
from models.fusion.cross_attention import CrossAttentionFusion
from models.classifier.mlp_classifier import MLPClassifier


class EnhancedMetadataEncoder(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=128, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Linear(output_dim, 768)

    def forward(self, features):
        x = self.encoder(features)
        x = self.projection(x)
        return x


class LeafMaskAttention(nn.Module):
    """
    Attention module sử dụng leaf mask để focus vào vùng lá
    KHÔNG nhân mask vào ảnh, mà dùng mask để tạo attention weights
    """
    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        # Convolution để xử lý mask thành attention map
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Feature projection
        self.feature_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
    def forward(self, features, leaf_mask):
        """
        Args:
            features: [B, C, H, W] - Feature maps từ vision encoder
            leaf_mask: [B, H, W] - Mask từ hull (1.0=leaf, 0.0=background)
        Returns:
            attended_features: [B, C, H, W] - Features đã được attention
            attention_weights: [B, 1, H, W] - Attention weights từ mask
        """
        # Resize mask về cùng kích thước feature maps
        if leaf_mask.shape[-2:] != features.shape[-2:]:
            mask_resized = F.interpolate(
                leaf_mask.unsqueeze(1),
                size=features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            mask_resized = leaf_mask.unsqueeze(1)
        
        # Encode mask thành attention weights
        attention_weights = self.mask_encoder(mask_resized)  # [B, 1, H, W]
        
        # Apply attention vào features
        attended_features = features * attention_weights
        
        return attended_features, attention_weights


class RiceDiseaseMultimodalModel(nn.Module):
    def __init__(self, num_classes=4, fusion_type="cross_attention", 
                 use_metadata=True, use_leaf_mask=True):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_metadata = use_metadata
        self.use_leaf_mask = use_leaf_mask

        # Vision branch - trả về feature maps
        self.vision_encoder = EfficientNetEncoder(return_feature_maps=True)
        self.vision_projection = VisionProjection(input_dim=1280, output_dim=768)

        # Text branch
        self.text_encoder = PhoBERTEncoder()
        self.text_projection = TextProjection(input_dim=768, output_dim=768)

        # Leaf mask attention (QUAN TRỌNG - KHÔNG NHÂN VÀO ẢNH)
        if self.use_leaf_mask:
            self.leaf_mask_attention = LeafMaskAttention(feature_dim=1280, hidden_dim=128)

        # Metadata branch
        if self.use_metadata:
            self.metadata_encoder = EnhancedMetadataEncoder(input_dim=26, hidden_dim=128, output_dim=256)

        # Fusion
        self.concat_fusion = ConcatFusion(image_dim=768, text_dim=768, output_dim=768)
        self.cross_attention = CrossAttentionFusion(embed_dim=768, num_heads=8)

        # Classifier
        self.classifier = MLPClassifier(input_dim=768, hidden_dim=512, num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask, metadata_features=None, leaf_mask=None):
        """
        Args:
            images: [B, 3, H, W] - Ảnh gốc (đã resize)
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            metadata_features: [B, 26]
            leaf_mask: [B, H, W] - Mask từ hull (1.0=leaf, 0.0=background)
                      KHÔNG nhân vào ảnh, mà dùng để tạo attention
        """
        # 1. Extract vision feature maps [B, C, H', W']
        visual_features = self.vision_encoder(images)
        
        # 2. ÁP DỤNG LEAF MASK ATTENTION - KHÔNG NHÂN VÀO ẢNH
        if self.use_leaf_mask and leaf_mask is not None:
            # Dùng mask để tạo attention weights
            attended_features, attention_weights = self.leaf_mask_attention(visual_features, leaf_mask)
            # Pooling sau attention
            visual_pooled = attended_features.mean(dim=[2, 3])
        else:
            # Global pooling bình thường
            visual_pooled = visual_features.mean(dim=[2, 3])
        
        visual_feat = self.vision_projection(visual_pooled)  # [B, 768]

        # 3. Extract text features
        text_feat = self.text_encoder(input_ids, attention_mask)  # [B, 768]
        text_feat = self.text_projection(text_feat)  # [B, 768]

        # 4. Fusion
        if self.fusion_type == "concat":
            fused = self.concat_fusion(visual_feat, text_feat)
        else:
            fused, _ = self.cross_attention(visual_feat, text_feat)

        # 5. Integrate metadata
        if self.use_metadata and metadata_features is not None:
            metadata_feat = self.metadata_encoder(metadata_features)
            fused = fused + metadata_feat

        # 6. Classification
        logits = self.classifier(fused)
        
        return logits