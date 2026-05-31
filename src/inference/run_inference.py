import argparse
from pathlib import Path
import json

import torch
import yaml
from transformers import AutoTokenizer
from PIL import Image

from datasets.transforms import get_val_transforms
from models.multimodal_model import RiceDiseaseMultimodalModel


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image(image_path, image_size):
    img = Image.open(image_path).convert("RGB")
    transform = get_val_transforms(image_size)
    return transform(img).unsqueeze(0)


def predict(
    image_path,
    text,
    checkpoint,
    config,
    device
):
    device = torch.device(device)

    model = RiceDiseaseMultimodalModel(
        num_classes=config["model"]["num_classes"],
        fusion_type=config["model"].get("fusion", "cross_attention"),
    )
    model.to(device)

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.get("text_tokenizer", "vinai/phobert-base"))

    image_tensor = load_image(image_path, config["data"]["image_size"]).to(device)

    if text is None:
        encoded = tokenizer(
            "",
            max_length=config["data"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(
            text,
            max_length=config["data"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())

    classes = config["dataset"]["classes"]
    result = {
        "prediction": classes[pred_idx],
        "index": pred_idx,
        "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default=None)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    config = load_config(args.config)

    out = predict(args.image, args.text, args.checkpoint, config, args.device)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
