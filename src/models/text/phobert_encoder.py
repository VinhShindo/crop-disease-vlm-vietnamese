import torch.nn as nn
from transformers import AutoModel, AutoConfig
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class PhoBERTEncoder(nn.Module):
    def __init__(
        self,
        model_name="vinai/phobert-base"
    ):
        super().__init__()

        # Load config trước để tránh lỗi
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model với config
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
        )

        self.output_dim = config.hidden_size

    def forward(
        self,
        input_ids,
        attention_mask
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0]
        return cls_embedding