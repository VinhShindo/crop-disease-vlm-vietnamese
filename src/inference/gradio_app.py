import gradio as gr
import torch
from pathlib import Path

from transformers import AutoTokenizer

from datasets.transforms import get_val_transforms
from models.multimodal_model import RiceDiseaseMultimodalModel


def build_app(model_path, config_path="configs/config.yaml", device="cpu"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    classes = config["dataset"]["classes"]
    tokenizer_name = config.get("text_tokenizer", "vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = RiceDiseaseMultimodalModel(
        num_classes=config["model"]["num_classes"],
        fusion_type=config["model"].get("fusion", "cross_attention"),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    transform = get_val_transforms(config["data"]["image_size"])

    def inference(image, text=""):
        if image is None:
            return "No image provided"

        img_path = Path(image.name)
        pil = image.convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(device)
        # gradio gives a PIL Image; apply transform directly
        import numpy as np
        if not hasattr(image, "convert"):
            return "Invalid image"
        pil = image.convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(device)

        encoded = tokenizer(
            text if text else "",
            max_length=config["data"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            logits = model(tensor, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())

        return classes[idx]

    iface = gr.Interface(
        fn=inference,
        inputs=[gr.Image(type="pil"), gr.Textbox(lines=2, placeholder="Optional text/context")],
        outputs=gr.Textbox(),
        title="Rice Disease Classifier",
    )
    return iface
