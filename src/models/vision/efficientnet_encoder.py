import timm
import torch.nn as nn


class EfficientNetEncoder(nn.Module):

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        self.output_dim = (
            self.backbone.num_features
        )

    def forward(self, images):

        features = self.backbone(images)

        return features