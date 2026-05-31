# TRAINING PIPELINE

## Phase 1 — Data Preparation

Dataset Cleaning

* kiểm tra ảnh lỗi
* kiểm tra metadata lỗi

---

Dataset Split

Train
Validation
Test

Recommended:

70%
15%
15%

Stratified Split

---

## Phase 2 — Image Processing

Resize

224×224

Normalize

ImageNet Statistics

Data Augmentation:

* Horizontal Flip
* Rotation
* Color Jitter
* Random Crop

---

## Phase 3 — Text Processing

Input:

Vietnamese metadata

Processing:

* lowercase
* tokenize
* attention mask

Tokenizer:

PhoBERT Tokenizer

---

## Phase 4 — Feature Extraction

Vision Encoder:

EfficientNet-B0

Text Encoder:

PhoBERT

---

## Phase 5 — Multimodal Fusion

Baseline:

Feature Concatenation

Research Model:

Cross-Attention Fusion

---

## Phase 6 — Classification

MLP Classifier

Output:

4 Disease Classes

---

## Phase 7 — Evaluation

Metrics:

* Accuracy
* Precision
* Recall
* Macro F1

Main Metric:

Macro F1-score

---

## Phase 8 — Explainable AI

Grad-CAM

Attention Visualization

Cross-Attention Analysis

---

## Phase 9 — Embedding Analysis

t-SNE

UMAP

Embedding Space Visualization

Mục tiêu:

* đánh giá semantic separation
* đánh giá multimodal alignment
