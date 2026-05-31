import torch.nn as nn

from transformers import (
    AutoModel
)


class PhoBERTEncoder(nn.Module):

    def __init__(
        self,
        model_name="vinai/phobert-base"
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        self.output_dim = 768

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = (
            outputs.last_hidden_state[:, 0]
        )

        return cls_embedding