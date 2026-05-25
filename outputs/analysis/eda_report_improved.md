# Improved Multimodal EDA — Rice Leaf Disease Dataset

This document is a research-grade exploratory data analysis describing the multimodal dataset, image/text alignment, dataset biases, and implications for multimodal model development. It is written for thesis and conference-readiness.

## 1. Executive summary
- Dataset: 3,355 high-resolution rice leaf images with Vietnamese textual metadata and structured fields (severity, weather, humidity, growth stage, farmer note).
- Classes: BrownSpot (523), LeafBlast (779), Hispa (565), Healthy (1,488).
- Primary concerns: class imbalance, synthetic/repetitive text metadata, limited environmental diversity, and potential image–text misalignment for some samples.

Implication: The dataset is a solid base for initial multimodal research, but any production or field-deployment claims require additional data collection and careful validation under real-world conditions.

## 2. Data provenance and capture modality
- Capture style: close-up leaf images on neutral backgrounds (consistent lighting), many images appear to be photographed under controlled conditions (paper or uniform background visible in samples).
- Date/region metadata: synthetic `location` fields in metadata (e.g., Nam Dinh, Long An) — useful for domain analysis but require provenance verification.

Risk: models trained on controlled, clean backgrounds typically overfit to photographic setup and underperform on field scenes with clutter, occlusion, and variable lighting.
 
## 3. Class distribution and sampling
- Healthy: 44.4% (1,488)
- LeafBlast: 23.2% (779)
- Hispa: 16.8% (565)
- BrownSpot: 15.6% (523)

Observations:
- Strong imbalance toward the Healthy class.
- Per-class sample counts are modest for deep learning (particularly BrownSpot and Hispa).

Immediate recommendations:
- Use stratified splits and K-fold cross-validation to report stable metrics.
- For final models, prefer Macro F1-score and per-class recall as primary evaluation criteria.
- Apply class rebalancing: weighted loss, focal loss, or targeted augmentation for minority classes.

## 4. Image quality and visual patterns
- Resolution: median ~2049×2049, enabling detailed texture-based analysis and patch-level modeling.
- Visual characteristics observed in samples:
  - Many images are leaf fragments placed on a uniform white background or tray — not natural field context.
  - Lesions, when present, are often small and localized; several disease classes share similar small-spot appearances.
  - Some samples (from manual inspection) show near-healthy leaves — potential label noise or subtle early-stage symptoms.

Actionable checks:
- Run per-image edge/contrast and color histogram analyses to flag images with low signal-to-noise for lesions.
- Create a lesion segmentation subset (manually label ~200 images across classes) to benchmark localization accuracy.

## 5. Text metadata analysis (Vietnamese descriptions)
- Each image contains multiple short Vietnamese descriptions (avg ~4 per image) and `symptoms` keywords.
- Linguistic characteristics: short, templated phrases such as "Lá xuất hiện các vùng tổn thương màu nâu" and "Triệu chứng đốm nâu phát triển mạnh" repeated across many records.

Problems identified:
- High redundancy: identical/similar sentences repeated across many images reduces textual diversity and may bias the text encoder toward templated patterns.
- Synthetic cues: structured fields (weather, humidity) appear plausible but are likely simulated; verify against acquisition logs.
- Mismatch risk: the visual evidence in a sampled image may not show the described lesion clearly (label–image mismatch), which undermines contrastive alignment or captioning training.

Quantitative checks to perform:
- Compute per-image text embedding variance (PhoBERT) to detect near-duplicate descriptions.
- Measure text-to-image retrieval precision on a hold-out set: given the texts, rank images and compute recall@K — low performance indicates weak alignment.

## 6. Image–text alignment insights
- When text is descriptive of obvious lesions (e.g., large necrotic patches), alignment is straightforward.
- For small or early-stage symptoms, text may contain claims (e.g., "severe") unsupported by the image crop.

Consequences for modeling:
- Training contrastive models (CLIP-style) with noisy pairs may learn incorrect associations; curriculum or noisy-label-robust losses recommended.
- Use a two-stage approach: (1) train contrastive model on high-confidence pairs (strong lesion visibility), (2) fine-tune multimodal fusion on full dataset.

## 7. Corrupted images and data hygiene
- The existing EDA reports no corrupted images; a CSV of corrupted images is present (outputs/analysis/corrupted_images.csv).
- Additional checks recommended: verify EXIF timestamps, remove near-duplicate frames (burst photos), and filter out images with extreme white balance or saturation.

## 8. Multimodal opportunities and suggested tasks
- Contrastive pretraining: adopt CLIP-style objectives with PhoBERT text embeddings to build a robust joint embedding space.
- Fine-grained cross-attention: fuse local image patches with token-level representations of Vietnamese phrases to capture lesion–phrase correspondence.
- Weakly supervised localization: use text tokens like "đốm nâu" to guide class activation mapping and generate pseudo-masks.
- Explainability: generate per-token and per-patch attribution maps; produce farmer-friendly explanations in Vietnamese.

## 9. Dataset limitations and risks (detailed)
- Domain gap: controlled backgrounds cause generalization failures in field settings (occlusion, soil, water droplets, complex backgrounds).
- Label noise: potential misalignment between severity assessments and visual severity.
- Text repetition: reduces the effective linguistic signal for the text encoder; harms semantic generalization.
- Class imbalance: leads to bias toward Healthy; recall for minority classes may be low.

For each issue we provide mitigation strategies in the companion document (dataset_quality_review.md).

## 10. Short experimental recipe (research)
1. Run image-only baseline (EfficientNet-B0) with strong augmentations (RandAugment, color jitter, random crops). Report Macro F1.
2. Train text-only baseline (PhoBERT) on labels from `texts` aggregated per-image (simple classifier head). Report Macro F1.
3. Train contrastive alignment (image encoder + PhoBERT) with cosine loss on high-confidence pairs (filter where text includes explicit lesion keywords and image lesion area > threshold).
4. Fine-tune a cross-attention fusion network using pre-trained encoders; evaluate on stratified validation folds.
5. Ablation: (a) remove text, (b) remove image, (c) remove structured metadata — report delta in Macro F1 and per-class recall.

## 11. Deliverables and recommended next steps
- Curate a high-confidence 500-image subset with verified alignment and lesion masks for robust pretraining and validation.
- Augment field images: collect 500 images from heterogeneous field conditions (different farms, lighting, occlusion).
- Re-run text curation: increase linguistic variety by paraphrasing, adding farmer-style notes, and including environmental context.
- Implement noisy-pair robust contrastive loss and curriculum fine-tuning.

---

For implementation details, reproducible scripts, and visualization notebooks see the `notebooks/` and `src/` folders. This report should be cited and used as the canonical EDA for academic write-ups.
