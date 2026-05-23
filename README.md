# `README.md`

```md
# Vision-Language Rice Leaf Disease Detection

## Overview

Đề tài xây dựng hệ thống Vision-Language AI cho nhận dạng sâu bệnh cây lúa từ:

- ảnh lá lúa
- mô tả hiện trường bằng tiếng Việt

Mục tiêu của đề tài là kết hợp Computer Vision và Natural Language Processing
để nâng cao khả năng nhận dạng bệnh cây trong môi trường nông nghiệp thông minh.

---

# Project Goals

- Nhận dạng bệnh lá lúa từ ảnh
- Kết hợp ảnh + text tiếng Việt
- Tăng độ chính xác nhờ multimodal learning
- Hỗ trợ nông nghiệp thông minh
- Hướng tới triển khai thực tế

---

# Main Architecture

## Vision Encoder
- EfficientNet-B0

## Text Encoder
- PhoBERT

## Fusion Strategy
- Cross Attention
- Feature Concatenation

## Output
- Disease Classification

---

# Disease Classes

Dataset hiện tại gồm 4 lớp:

- BrownSpot
- Healthy
- Hispa
- LeafBlast

---

# Recommended Metrics

## Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score

Main metric:
- F1-score

Lý do:
Dataset có hiện tượng imbalance giữa các lớp nên F1-score phản ánh hiệu quả tốt hơn Accuracy.

---

# Dataset Structure

dataset/
├── raw/
│   ├── BrownSpot/
│   ├── Healthy/
│   ├── Hispa/
│   └── LeafBlast/
│
└── text_metadata/

---

# Project Structure

CROP-DISEASE-VLM-VIET/
│
├── configs/
├── dataset/
├── docs/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore

---

# Recommended Hardware

## Minimum
- RTX 3060 12GB
- RAM 32GB

## Recommended
- RTX 4070
- RTX 4080
- RAM 32GB+

---

# EDA (Exploratory Data Analysis)

## Dataset Statistics

| Class | Number of Images |
|---|---|
| BrownSpot | 523 |
| Healthy | 1488 |
| Hispa | 565 |
| LeafBlast | 779 |

Total classes:
- 4

---

# Dataset Insights

## 1. Dataset Imbalance

Class Healthy có số lượng ảnh lớn hơn đáng kể so với các lớp bệnh.

Điều này có thể gây:
- model bias
- over-predict Healthy
- giảm Recall của lớp bệnh

### Giải pháp đề xuất
- Weighted Loss
- Data Augmentation
- Stratified Split
- F1-score monitoring

---

## 2. Dataset Quality

Ưu điểm:
- ảnh sạch
- background đơn giản
- ít noise
- object rõ ràng

Nhược điểm:
- chưa phản ánh điều kiện ngoài thực tế
- thiếu ảnh ánh sáng phức tạp
- thiếu ảnh ngoài đồng ruộng

---

## 3. Resolution Consistency

Resolution ảnh khá đồng đều.

Điều này phù hợp cho:
- EfficientNet
- ViT
- CNN training

Recommended resize:
- 224x224
- 256x256

---

# EDA Outputs

## Class Distribution

![Class Distribution](outputs/visualizations/class_distribution.png)

---

## Dataset Overview

![Dataset Overview](outputs/visualizations/dataset_overview.png)

---

## Image Resolution Distribution

![Resolution Distribution](outputs/visualizations/image_resolution_distribution.png)

---

# Current EDA Features

- Class distribution analysis
- Dataset visualization
- Resolution analysis
- Corrupted image detection
- Automatic EDA report generation

---

# Future Work

## Dataset
- Thu thập ảnh thực tế ngoài đồng
- Tăng diversity dữ liệu
- Tạo metadata tiếng Việt

## AI Model
- Vision-Language Fusion
- Segmentation
- Explainable AI
- Mobile deployment

## System
- Chatbot nông nghiệp
- Real-time inference
- Edge AI deployment

---

# Research Direction

Đề tài hướng tới:
- Multimodal AI
- Agricultural AI
- Vietnamese Vision-Language Models
- Smart Farming Systems