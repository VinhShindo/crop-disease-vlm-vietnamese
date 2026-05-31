**Configuration Reference — configs/config.yaml**

- **project_name**: Project identifier.

- **model**:
  - `image_encoder`: backbone name (e.g., `efficientnet_b0` via `timm`).
  - `text_encoder`: text model name (for documentation only; model class loads `vinai/phobert-base`).
  - `num_classes`: number of target classes.
  - `fusion`: `cross_attention` or `concat`.

- **dataset**:
  - `num_classes`: duplicate of `model.num_classes` for clarity.
  - `classes`: ordered list of class names (must match labels used in metadata JSON).

- **data**:
  - `train_metadata`, `val_metadata`, `test_metadata`: paths to JSON metadata files containing samples with keys: `image` (path), `label` (one of the `dataset.classes`), and optional `texts`, `symptoms`, `visual_analysis`, plus metadata fields like `humidity`, `temperature`, `severity`, `growth_stage`, `location`, `farmer_note`.
  - `image_size`: image short/long side size used for transforms.
  - `max_length`: tokenizer maximum token length.
  - `num_workers`: dataloader workers.

- **training**:
  - `seed`: random seed for deterministic runs.
  - `batch_size`: training/validation batch size.
  - `epochs`: maximum epochs.
  - `lr`: initial learning rate.
  - `weight_decay`: optimizer weight decay.
  - `grad_clip_norm`: gradient clipping value.
  - `patience`: early stopping patience (in epochs).

- **evaluation**:
  - `metrics`: metrics to compute after evaluation.

- **outputs**:
  - `checkpoints`: directory to save models.
  - `figures`: directory to save visualizations (confusion matrices, GradCAM outputs).
  - `metrics`: directory to save JSON metric summaries.
  - `predictions`: directory for saved per-sample predictions.

Notes and recommendations
- Keep the `dataset.classes` ordering stable: it determines the mapping between model outputs and human-readable labels.
- Metadata fields are optional but if present the dataset loader exposes them in `sample['metadata']` for future multi-input models.
- Example minimal config: see `configs/config.yaml` in the repo.
