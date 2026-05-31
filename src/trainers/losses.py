import torch
import torch.nn as nn


def build_loss(
    class_weights=None
):

    if class_weights is not None:

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion