import torch.nn as nn

from models.vision.efficientnet_encoder import EfficientNetEncoder
from models.vision.vision_projection import VisionProjection
from models.text.phobert_encoder import PhoBERTEncoder
from models.text.text_projection import TextProjection
from models.fusion.concat_fusion import ConcatFusion
from models.fusion.cross_attention import CrossAttentionFusion
from models.classifier.mlp_classifier import MLPClassifier
from models.metadata_encoder import MetadataEncoder


class RiceDiseaseMultimodalModel(nn.Module):

    def __init__(self, num_classes=4, fusion_type="cross_attention", use_metadata=False):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_metadata = use_metadata

        # ----------------
        # Vision
        # ----------------
        self.vision_encoder = EfficientNetEncoder()
        self.vision_projection = VisionProjection(input_dim=1280, output_dim=768)

        # ----------------
        # Text
        # ----------------
        self.text_encoder = PhoBERTEncoder()
        self.text_projection = TextProjection(input_dim=768, output_dim=768)

        # ----------------
        # Metadata
        # ----------------
        self.metadata_encoder = MetadataEncoder(input_dim=16, hidden_dim=64, output_dim=128)

        # ----------------
        # Fusion
        # ----------------
        self.concat_fusion = ConcatFusion()
        self.cross_attention = CrossAttentionFusion()

        # ----------------
        # Classifier
        # ----------------
        self.classifier = MLPClassifier(num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask, metadata_features=None):
        image_feat = self.vision_encoder(images)
        image_feat = self.vision_projection(image_feat)

        text_feat = self.text_encoder(input_ids, attention_mask)
        text_feat = self.text_projection(text_feat)

        if self.use_metadata and metadata_features is not None:
            metadata_feat = self.metadata_encoder(metadata_features)
            image_feat = image_feat + metadata_feat.unsqueeze(1)

        if self.fusion_type == "concat":
            fused = self.concat_fusion(image_feat, text_feat)
        else:
            fused, _ = self.cross_attention(image_feat, text_feat)

        logits = self.classifier(fused)
        return logits
