# Modeling Recommendations for Vietnam Rice Leaf Disease VLM

## 1. Fix the metadata repetition problem

The current dataset uses only 32 unique text templates for 3,355 samples and 12 unique visual analysis phrases. This is a strong signal that metadata enrichment is required.

### Action items

- Add at least **5 unique sentence templates per disease class**.
- Introduce **farmer-style phrasing** like:
  - "Lá xuất hiện đốm nâu nhỏ như vết cháy giữa phiến lá."
  - "Vệt trắng dài do Hispa làm lá mất đi độ bóng tự nhiên."
  - "Mảng bệnh màu xám vàng chạy dọc theo sống lá."
- Include **severity tags**: mild/moderate/severe.

## 2. Improve image-text alignment

The current `image_text_pairs.png` artifact is broken and needs a real paired display. For research-grade evaluation, include:

- at least 50 manually verified image-text pairs
- retrieval examples for query text -> top-k image
- retrieval examples for image -> top-k text

## 3. Upgrade explainability

The pseudo attention heatmap lacks interpretability. Replace it with:

- Grad-CAM on the current CNN backbone
- attention map overlays for Vietnamese symptom tokens
- comparison of correct vs incorrect predictions

## 4. Address class imbalance

Healthy dominates the dataset at 44.4%. To mitigate imbalance:

- use weighted cross-entropy or focal loss
- oversample minority disease classes
- apply class-specific augmentations for BrownSpot, Hispa, LeafBlast

## 5. Improve research visualization quality

Add the following charts to support thesis-level reporting:

- training/validation curves for accuracy and loss
- per-class precision/recall bar plots
- multimodal retrieval heatmaps
- hardest-sample gallery for failure analysis

## 6. Recommended model pipeline

- Stage 1: train image encoder + contrastive loss with high-confidence pairs
- Stage 2: fine-tune with cross-attention fusion using full dataset
- Stage 3: evaluate using Macro F1, per-class recall, and retrieval metrics

## 7. Future dataset expansion

Add field-collected images with:

- irregular backgrounds
- mixed lighting
- partial leaves and occlusion
- farmer observation notes

This will make the dataset more robust for real agricultural deployment.
