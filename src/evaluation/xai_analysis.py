from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path
import torch


def generate_gradcam(model, input_tensor, target_layer, output_path, device="cpu"):
    model.to(device)
    model.eval()

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device=="cuda"))
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = input_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(visualization).save(str(output_path))
    return str(output_path)
