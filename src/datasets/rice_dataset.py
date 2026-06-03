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

# Mapping cho categorical features
WEATHER_MAP = {"sunny": 0, "cloudy": 1, "rainy": 2}
SEVERITY_MAP = {"low": 0, "moderate": 1, "high": 2}
GROWTH_STAGE_MAP = {"seedling": 0, "tillering": 1, "booting": 2, "flowering": 3, "ripening": 4}
LOCATION_MAP = {"Dong Thap": 0, "An Giang": 1, "Can Tho": 2, "Long An": 3, "other": 4}

METADATA_FIELDS = [
    "weather", "humidity", "temperature", "severity", 
    "growth_stage", "location", "farmer_note",
    "leaf_area_ratio", "background_ratio", "lesion_area_ratio", "annotation_confidence"
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
        deterministic_text=True,
        use_metadata=True,  # Thêm flag
    ):
        self.metadata_path = Path(metadata_path)
        self.transform = transform
        self.max_length = max_length
        self.use_symptoms = use_symptoms
        self.use_visual_analysis = use_visual_analysis
        self.metadata_fields = metadata_fields or METADATA_FIELDS
        self.deterministic_text = deterministic_text
        self.use_metadata = use_metadata

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.deterministic_text:
            self.processed_texts = [self._build_text(sample) for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def _build_text(self, sample):
        texts = sample.get("texts", [])
        symptoms = sample.get("symptoms", [])
        visual_analysis = sample.get("visual_analysis", [])

        text_parts = []
        
        if texts:
            text_parts.extend(texts)
            
        if self.use_symptoms and symptoms:
            text_parts.extend(symptoms)
            
        if self.use_visual_analysis and visual_analysis:
            text_parts.extend(visual_analysis)
        
        separator = "\n[SEP]\n"
        return separator.join(text_parts).strip()

    def _encode_metadata(self, sample):
        """Encode metadata thành tensor features (22 dimensions)"""
        features = []
        
        # 1. Weather (one-hot, 3 dims)
        weather = sample.get("weather", "sunny")
        weather_onehot = torch.zeros(3)
        weather_onehot[WEATHER_MAP.get(weather, 0)] = 1
        features.append(weather_onehot)
        
        # 2. Humidity (normalized, 1 dim)
        humidity = sample.get("humidity", 70)
        features.append(torch.tensor([humidity / 100.0]))
        
        # 3. Temperature (normalized 15-35°C, 1 dim)
        temp = sample.get("temperature", 25)
        features.append(torch.tensor([(temp - 15) / 20.0]))
        
        # 4. Severity (one-hot, 3 dims)
        severity = sample.get("severity", "moderate")
        severity_onehot = torch.zeros(3)
        severity_onehot[SEVERITY_MAP.get(severity, 1)] = 1
        features.append(severity_onehot)
        
        # 5. Growth stage (one-hot, 5 dims)
        growth = sample.get("growth_stage", "tillering")
        growth_onehot = torch.zeros(5)
        growth_onehot[GROWTH_STAGE_MAP.get(growth, 1)] = 1
        features.append(growth_onehot)
        
        # 6. Location (one-hot, 5 dims)
        location = sample.get("location", "Dong Thap")
        location_onehot = torch.zeros(5)
        location_onehot[LOCATION_MAP.get(location, 4)] = 1
        features.append(location_onehot)
        
        # 7. Farmer note length (normalized, 1 dim)
        farmer_note = sample.get("farmer_note", "")
        features.append(torch.tensor([min(len(farmer_note) / 200.0, 1.0)]))
        
        # 8. Leaf area ratios (3 dims)
        leaf_area = sample.get("leaf_area_ratio", 0.1)
        features.append(torch.tensor([min(leaf_area, 1.0)]))
        background_area = sample.get("background_ratio", 0.9)
        features.append(torch.tensor([min(background_area, 1.0)]))
        lesion_area = sample.get("lesion_area_ratio", 0.05)
        features.append(torch.tensor([min(lesion_area, 1.0)]))
        
        # 9. Annotation confidence (1 dim)
        confidence = sample.get("annotation_confidence", 0.8)
        features.append(torch.tensor([confidence]))
        
        # Concatenate tất cả features (3+1+1+3+5+5+1+3+1 = 23 dims)
        metadata_tensor = torch.cat(features)
        return metadata_tensor

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image_path = Path(sample["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Process text
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

        # Process metadata
        metadata_tensor = None
        if self.use_metadata:
            metadata_tensor = self._encode_metadata(sample)

        label = LABEL2ID[sample["label"]]

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": text,
            "image_path": str(image_path),
            "metadata": metadata_tensor,  # Bây giờ là tensor
        }