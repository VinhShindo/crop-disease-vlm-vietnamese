Inference usage

1) Quick CLI

Run the lightweight inference CLI to predict a single image (optionally add a short text context):

```bash
python src/inference/run_inference.py --image path/to/image.jpg --text "lá có vết" --checkpoint outputs/checkpoints/best_model.pth --device cpu
```

Output (JSON): prediction label, index, and per-class probabilities.

2) Gradio app (web demo)

Start the Gradio demo (loads model from checkpoint):

```python
from inference.gradio_app import build_app

app = build_app(model_path="outputs/checkpoints/best_model.pth", device="cpu")
app.launch()
```

Notes
- Ensure `configs/config.yaml` points to correct `dataset.classes` and metadata paths.
- If using GPU, pass `--device cuda` or `device="cuda"` when launching Gradio.
- The inference runner uses the same transforms and tokenizer settings defined in the repo for consistency.
