
# Vision-Language Rice Leaf Disease Detection

## Overview

Đề tài xây dựng hệ thống Vision-Language AI cho nhận dạng sâu bệnh cây lúa
từ:
- ảnh lá lúa
- mô tả hiện trường bằng tiếng Việt

## Goal

- Nhận dạng bệnh lá lúa
- Kết hợp ảnh + text tiếng Việt
- Hỗ trợ nông nghiệp thông minh

---

# Main Architecture

## Vision Encoder
- EfficientNet-B0

## Text Encoder
- PhoBERT

## Fusion
- Cross Attention / Concatenation

## Output
- Disease Classification

---

# Disease Classes

- healthy
- blast
- brown_spot
- bacterial_blight

---

# Recommended Metrics

- Accuracy
- Precision
- Recall
- F1-score

Main metric:
- F1-score

---

# Dataset Structure

dataset/
├── images/
├── metadata/
└── annotations/

---

# Project Structure

- docs/
- src/
- outputs/
- scripts/

---

# Recommended Hardware

Minimum:
- RTX 3060 12GB
- RAM 32GB

Recommended:
- RTX 4070
- RAM 32GB+

---

# Future Work

- Segmentation
- Mobile deployment
- Chatbot nông nghiệp
- Real-time inference

