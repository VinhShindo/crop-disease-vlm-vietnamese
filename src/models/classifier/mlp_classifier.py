import torch.nn as nn


class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=512,
        num_classes=4
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(self, x):
        return self.classifier(x)