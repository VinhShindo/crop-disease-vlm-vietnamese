# Dataset Quality Review — Rice Leaf Disease Multimodal Dataset

This document provides a technical dataset quality assessment, metadata improvement suggestions, training recommendations, multimodal research directions, weakness analysis, and future work — written for thesis-level documentation.

## 1. Summary assessment
- Overall: dataset is well-structured and rich in multimodal signals but contains several issues that must be addressed before claiming field-ready performance.
- Strengths: high-resolution images, multimodal metadata (Vietnamese descriptions), clear disease taxonomy, reproducible repository structure.
- Key weaknesses: class imbalance, synthetic/repetitive text generation, potential image–text misalignment, limited environmental diversity.

## 2. Detailed dataset-quality findings

2.1 Class balance
- Healthy dominates (~44%). BrownSpot and Hispa are under-represented.
- Effect: training instability, classifier bias, low minority-class recall.
- Fixes: targeted augmentation, oversampling, class-weighted loss, synthetic minority augmentation only when visually plausible.

2.2 Image–label alignment and label noise
- Manual spot-checks indicate some images labelled as diseased have no obvious lesions at the crop used by the classifier (small crops/close-ups occasionally miss lesions).
- Severity labels appear programmatic and sometimes inconsistent with visual severity.
- Fixes:
  - Create a validation subset where an agronomist or trained annotator confirms lesion presence and severity.
  - If budget allows, annotate bounding boxes or segmentation masks for lesions on a representative subset (200–500 images).
  - Use automated filtering: compute color/texture anomaly scores and compare to labels to flag suspicious pairs.

2.3 Text metadata realism and diversity
- Observed repeated templated sentences across many records (e.g., "Triệu chứng đốm nâu phát triển mạnh").
- Risk: text encoder will learn dataset-specific templates and fail to generalize to farmer reports.
- Fixes:
  - Enrich texts with more natural language variety: paraphrase with language models (careful human vetting), add contextual sentences (soil, irrigation, pesticide history), and include farmer-observed time series ("triệu chứng xuất hiện 3 ngày trước").
  - Add negative statements where appropriate ("không có vết thối").
  - Retain and surface `farmer_note` style sentences rather than only templated symptom phrases.

2.4 Environmental diversity & domain gap
- Many images use uniform backgrounds and controlled lighting — not representative of in-situ field photos.
- Fixes:
  - Data collection: add field-captured images across different farms, seasons, and weather.
  - Synthetic augmentation: photometric transforms and background composite augmentation to simulate field scenes.
  - Domain adaptation: use style-transfer or fine-tune with small field-captured validation set.

2.5 Structured metadata (weather, humidity, temperature)
- Potentially synthetic values; useful if accurate but dangerous if incorrect.
- Fixes:
  - Attach provenance for structured fields (sensor, manual, simulated).
  - If simulated, mark them as synthetic so models treat them appropriately (don’t overfit to synthetic correlations).

## 3. Metadata-improvement concrete plan
1. Add a `confidence` field to each metadata record indicating whether the image–text pair was human-verified (`high`, `medium`, `low`).
2. Build a corrected subset of 500 `high` confidence records (manual review), to be used for contrastive pretraining and evaluation.
3. Expand `texts` diversity:
   - For each image, ensure at least two farmer-style sentences and two clinical symptom sentences.
   - Use controlled paraphrasing with PhoBERT-guided augmentation and human vetting.
4. Normalize severity into numeric scale (0–5) alongside categorical labels to enable regression-style severity estimation.
5. Add `capture_context` (camera distance, background type: lab/field, lighting condition) to model domain shift explicitly.

## 4. Training and modeling recommendations

4.1 Preprocessing and augmentation
- Image: resize to 256 then random crop to 224 for training; apply RandomResizedCrop, color jitter, Gaussian blur, MixUp/CutMix for regularization.
- Text: Vietnamese tokenization with PhoBERT tokenizer; minimal aggressive truncation (most texts are short).
- Structured metadata: standardize and normalize numeric fields.

4.2 Baselines and advanced strategies
- Baselines
  - Vision-only: EfficientNet-B0 + classifier head.
  - Text-only: PhoBERT + classifier head using aggregated `texts`.
  - Multimodal: pretrained EfficientNet + PhoBERT with cross-attention fusion layer.

- Advanced
  - CLIP-style contrastive pretraining (image encoder vs. PhoBERT embeddings) on `high` confidence pairs.
  - Hard negative mining: treat different images from same field/time as negatives for stronger discrimination.
  - Multi-task head: classification + severity regression + localization (weakly supervised CAM loss).
  - Use focal loss or class-balanced loss to address imbalance.

4.3 Regularization & generalization
- Label smoothing, dropout in projection heads, weight decay, and stochastic depth for transformer backbones.
- Early stopping on Macro F1 and maintain a balanced validation fold.

4.4 Evaluation practice
- Report Macro F1 as primary metric.
- Provide per-class precision/recall/F1 and confusion matrices.
- Report retrieval metrics for multimodal alignment: text->image recall@1/5/10 and image->text recall@1/5/10.
- Visualize attention maps and per-token attributions for explainability; include Vietnamese summaries of model rationales.

## 5. Multimodal research recommendations
- Contrastive + generative hybrid: train joint encoder with contrastive loss and a lightweight decoder for caption refinement in Vietnamese.
- Cross-modal localization: learn to predict pixel-level lesion maps guided by tokens ("đốm nâu", "vết cháy") using weak supervision.
- Semi-supervised learning: leverage large unlabelled field images and pseudo-label with model ensembles.
- Transfer learning: pretrain image encoder on plant disease datasets and PhoBERT on agricultural corpora (farmer forums, extension service notes).

## 6. Weakness analysis — impact and fixes
- Repetitive metadata
  - Why problematic: reduces effective language signal; text encoder learns dataset-specific templates leading to poor transfer.
  - How to fix: increase diversity via paraphrasing, collect farmer-written notes, add negative/contrastive sentences.
  - Expected effect: improves retrieval and alignment, yields more informative text supervision.

- Image–text mismatch
  - Why problematic: corrupts contrastive objectives; classifier may learn spurious correlations.
  - How to fix: curate high-confidence subset, introduce noisy-label-robust losses, and add lesion masks for supervision.
  - Expected effect: improved alignment and reduced false positives in deployment.

- Class imbalance
  - Why problematic: low recall on minority diseases; agricultural risk (missed disease warnings) is high.
  - How to fix: oversample disease classes, use weighted/focal loss, and collect more disease images.
  - Expected effect: higher per-class recall and more reliable field alerts.

## 7. Future work and roadmap (12 months)
- Months 0–2: Create `high` confidence subset (500 images) with manual verification; generate lesion bounding boxes for 200 images.
- Months 2–4: Implement contrastive pretraining on high-confidence subset; run initial cross-attention fusion experiments.
- Months 4–8: Field data collection (500–1,000 images) and domain adaptation experiments; expand Vietnamese text diversity.
- Months 8–12: Robust evaluation, calibration, and explainability prototypes; prepare manuscript and dataset release with documentation and data-use agreement.

---

Contact: for follow-up annotation work, reproducible experiment scripts, or to request the high-confidence review spreadsheet, open an issue or contact the maintainer in the repository.
