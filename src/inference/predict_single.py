import torch
from PIL import Image
from pathlib import Path

from datasets.rice_dataset import RiceDiseaseDataset
from datasets.transforms import get_val_transforms


def load_image(image_path, image_size=224):
    img = Image.open(image_path).convert("RGB")
    transform = get_val_transforms(image_size)
    return transform(img).unsqueeze(0)


def predict(model, image_path, tokenizer_max_length=128, device="cpu"):
    model.to(device)
    model.eval()

    img_tensor = load_image(image_path)
    img_tensor = img_tensor.to(device)

    # dummy text inputs (model expects text tensors)
    dummy_input_ids = torch.zeros((1, tokenizer_max_length), dtype=torch.long).to(device)
    dummy_attention = torch.zeros((1, tokenizer_max_length), dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, dummy_input_ids, dummy_attention)
        preds = outputs.argmax(dim=1).cpu().numpy()

    return int(preds[0])
