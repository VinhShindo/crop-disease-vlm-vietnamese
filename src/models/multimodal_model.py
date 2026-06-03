import torch
import torch.nn as nn

from models.vision.efficientnet_encoder import EfficientNetEncoder
from models.vision.vision_projection import VisionProjection
from models.text.phobert_encoder import PhoBERTEncoder
from models.text.text_projection import TextProjection
from models.fusion.concat_fusion import ConcatFusion
from models.fusion.cross_attention import CrossAttentionFusion
from models.classifier.mlp_classifier import MLPClassifier


class EnhancedMetadataEncoder(nn.Module):
    """Enhanced metadata encoder for 23-dim input"""
    def __init__(self, input_dim=23, hidden_dim=128, output_dim=256):
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


class RiceDiseaseMultimodalModel(nn.Module):
    def __init__(self, num_classes=4, fusion_type="cross_attention", use_metadata=True):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_metadata = use_metadata

        # Vision branch
        self.vision_encoder = EfficientNetEncoder()
        self.vision_projection = VisionProjection(input_dim=1280, output_dim=768)

        # Text branch
        self.text_encoder = PhoBERTEncoder()
        self.text_projection = TextProjection(input_dim=768, output_dim=768)

        # Metadata branch
        if self.use_metadata:
            self.metadata_encoder = EnhancedMetadataEncoder(input_dim=23, hidden_dim=128, output_dim=256)

        # Fusion
        self.concat_fusion = ConcatFusion(image_dim=768, text_dim=768, output_dim=768)
        self.cross_attention = CrossAttentionFusion(embed_dim=768, num_heads=8)

        # Classifier
        self.classifier = MLPClassifier(input_dim=768, hidden_dim=512, num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask, metadata_features=None):
        # Extract features
        image_feat = self.vision_encoder(images)      # [B, 1280]
        image_feat = self.vision_projection(image_feat)  # [B, 768]

        text_feat = self.text_encoder(input_ids, attention_mask)  # [B, 768]
        text_feat = self.text_projection(text_feat)  # [B, 768]

        # Fusion image + text
        if self.fusion_type == "concat":
            fused = self.concat_fusion(image_feat, text_feat)  # [B, 768]
        else:
            # Cross attention - function expects 2D tensors
            fused, _ = self.cross_attention(image_feat, text_feat)  # [B, 768]

        # Integrate metadata if available
        if self.use_metadata and metadata_features is not None:
            metadata_feat = self.metadata_encoder(metadata_features)  # [B, 768]
            fused = fused + metadata_feat  # Add metadata features

        logits = self.classifier(fused)  # [B, num_classes]
        return logits