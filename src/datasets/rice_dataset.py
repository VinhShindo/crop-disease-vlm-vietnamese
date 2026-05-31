import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


LABEL2ID = {
    "BrownSpot": 0,
    "Healthy": 1,
    "Hispa": 2,
    "LeafBlast": 3,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

METADATA_FIELDS = [
    "weather",
    "humidity",
    "temperature",
    "severity",
    "growth_stage",
    "location",
    "farmer_note",
]


class RiceDiseaseDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        tokenizer_name="vinai/phobert-base",
        max_length=128,
        transform=None,
        use_symptoms=True,
        use_visual_analysis=True,
        metadata_fields=None,
        deterministic_text=True,  # Thêm để fix random
    ):
        self.metadata_path = Path(metadata_path)
        self.transform = transform
        self.max_length = max_length
        self.use_symptoms = use_symptoms
        self.use_visual_analysis = use_visual_analysis
        self.metadata_fields = metadata_fields or METADATA_FIELDS
        self.deterministic_text = deterministic_text

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Preprocess text để deterministic
        if self.deterministic_text:
            self.processed_texts = [self._build_text(sample) for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def _build_text(self, sample):
        texts = sample.get("texts", [])
        symptoms = sample.get("symptoms", [])
        visual_analysis = sample.get("visual_analysis", [])

        text_parts = []
        
        # Không dùng random nữa, lấy text đầu tiên hoặc join tất cả
        if texts:
            text_parts.append(texts[0])  # Lấy text đầu tiên thay vì random

        if self.use_symptoms and symptoms:
            text_parts.extend(symptoms[:3])

        if self.use_visual_analysis and visual_analysis:
            text_parts.extend(visual_analysis[:2])

        return " ".join(text_parts).strip()

    def _encode_metadata(self, metadata):
        """Encode metadata thành tensor features"""
        # TODO: Implement metadata encoding
        # Tạm thời return None
        return None

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = Path(sample["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Lấy text đã được preprocess hoặc build mới
        if self.deterministic_text:
            text = self.processed_texts[idx]
        else:
            text = self._build_text(sample)
            
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        metadata = {
            field: sample.get(field)
            for field in self.metadata_fields
            if sample.get(field) is not None
        }

        label = LABEL2ID[sample["label"]]

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": text,
            "image_path": str(image_path),
            "metadata": metadata,
        }