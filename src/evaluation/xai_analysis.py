# evaluation/xai_analysis.py - FIXED VERSION

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import warnings

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch_grad_cam not installed. Install with: pip install pytorch_grad_cam")


def generate_gradcam_visualization(model, dataloader, device, class_names, output_dir="outputs", max_samples=10):
    """Generate GradCAM visualizations for correct and incorrect predictions - FIXED version"""
    if not GRADCAM_AVAILABLE:
        print("GradCAM not available. Skipping...")
        print("Install with: pip install pytorch_grad_cam")
        return
    
    model.eval()
    
    # Target layer for GradCAM
    if hasattr(model.vision_encoder.backbone, 'blocks'):
        target_layers = [model.vision_encoder.backbone.blocks[-1]]
    elif hasattr(model.vision_encoder.backbone, 'features'):
        target_layers = [model.vision_encoder.backbone.features[-1]]
    else:
        print("Could not find suitable target layer for GradCAM. Skipping...")
        return
    
    output_path = Path(output_dir) / "gradcam"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get samples
    samples_to_visualize = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_samples:
            break
        samples_to_visualize.append(batch)
    
    print(f"Generating GradCAM for {len(samples_to_visualize)} batches...")
    
    for batch_idx, batch in enumerate(samples_to_visualize):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask_text"].to(device)  # FIXED: use correct key
        labels = batch["label"].to(device)
        metadata = batch.get("metadata")
        
        if metadata is not None and isinstance(metadata, torch.Tensor):
            metadata = metadata.to(device)
        
        # Forward pass
        outputs = model(images, input_ids, attention_mask, metadata)
        preds = outputs.argmax(dim=1)
        
        # Generate GradCAM for each image
        for i in range(min(len(images), 3)):
            try:
                cam = GradCAM(model=model.vision_encoder.backbone, target_layers=target_layers)
                
                input_tensor = images[i].unsqueeze(0)
                grayscale_cam = cam(input_tensor=input_tensor)[0]
                
                # Denormalize image
                img_np = images[i].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
                
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                
                true_label = class_names[labels[i].item()]
                pred_label = class_names[preds[i].item()]
                status = "correct" if labels[i] == preds[i] else "incorrect"
                
                save_name = f"batch{batch_idx}_img{i}_{status}_true_{true_label}_pred_{pred_label}.png"
                save_path = output_path / save_name
                Image.fromarray(visualization).save(save_path)
                
            except Exception as e:
                print(f"Error generating GradCAM for image {batch_idx}_{i}: {e}")
                continue
    
    print(f"✓ GradCAM visualizations saved to {output_path}")
    print(f"  Generated {len(list(output_path.glob('*.png')))} images")