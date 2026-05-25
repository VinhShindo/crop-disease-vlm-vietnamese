# 🌾 Vision-Language Rice Leaf Disease Detection

## 📌 Overview

Đề tài tập trung xây dựng hệ thống AI đa phương thức (Multimodal AI)
cho bài toán nhận dạng sâu bệnh cây lúa bằng cách kết hợp:

- 🖼️ Ảnh lá lúa
- 📝 Mô tả hiện trường bằng tiếng Việt

Hệ thống hướng tới việc kết hợp:
- Computer Vision
- Natural Language Processing
- Vision-Language Learning

nhằm nâng cao khả năng nhận dạng bệnh cây trong môi trường nông nghiệp thông minh.

---

# 🎯 Project Goals

## Mục tiêu chính

- ✅ Nhận dạng bệnh lá lúa từ ảnh
- ✅ Kết hợp ảnh + văn bản tiếng Việt
- ✅ Tăng độ chính xác nhờ multimodal learning
- ✅ Giảm nhầm lẫn giữa các bệnh có triệu chứng giống nhau
- ✅ Hỗ trợ nông nghiệp thông minh
- ✅ Hướng tới triển khai thực tế

---

# 🧠 Problem Statement

Bài toán không chỉ là Image Classification truyền thống.

Hệ thống cần:
- hiểu đặc trưng hình ảnh của lá bệnh
- hiểu ngữ nghĩa triệu chứng bằng tiếng Việt
- học sự liên kết giữa ảnh và văn bản

Ví dụ:

```text
Ảnh:
Lá lúa xuất hiện nhiều đốm nâu.

Text:
"Trời mưa kéo dài, lá xuất hiện các đốm nâu nhỏ."

=> Predict:
BrownSpot
```

---

# 🚀 Research Motivation

Trong thực tế:

- nhiều bệnh có triệu chứng hình ảnh gần giống nhau
- điều kiện môi trường ảnh hưởng mạnh đến bệnh cây
- chỉ dùng ảnh có thể gây nhầm lẫn

Do đó:
- văn bản mô tả hiện trường
- triệu chứng bệnh
- thông tin ngữ cảnh

sẽ giúp AI hiểu bệnh tốt hơn.

---

# 🧬 Main Architecture

## 🔍 Vision Encoder

### EfficientNet-B0

Sử dụng để:
- trích xuất đặc trưng hình ảnh lá lúa
- học texture
- học vùng tổn thương
- nhận biết pattern bệnh

### Advantages

- nhẹ
- hiệu quả cao
- phù hợp transfer learning
- tối ưu cho GPU tầm trung

---

## 📝 Text Encoder

### PhoBERT

Sử dụng để:
- xử lý tiếng Việt
- học semantic disease descriptions
- hiểu mô tả triệu chứng bệnh

### Advantages

- pretrained tiếng Việt
- semantic embedding mạnh
- phù hợp NLP tiếng Việt

---

## 🔗 Fusion Strategy

### Feature Concatenation

- đơn giản
- ổn định
- dễ train

### Cross Attention

Cho phép:
- text attention vào vùng bệnh trên ảnh
- học image-text semantic alignment

---

## 🎯 Output

### Disease Classification

Predict:
- loại bệnh
- tình trạng lá lúa

---

# 🌱 Disease Classes

Dataset hiện tại gồm 4 lớp:

| Disease | Description |
|---|---|
| BrownSpot | Bệnh đốm nâu |
| Healthy | Lá khỏe mạnh |
| Hispa | Sâu hại Hispa |
| LeafBlast | Đạo ôn lá |

---

# 🗂️ Dataset Overview

## 📦 Dataset Summary

| Item | Value |
|---|---|
| Total Images | 3355 |
| Total Classes | 4 |
| Average Resolution | 2049 × 2049 |
| Metadata | Vietnamese descriptions |
| Dataset Type | Multimodal |

---

# 🧾 Dataset Structure

```text
dataset/
├── raw/
│   ├── BrownSpot/
│   ├── Healthy/
│   ├── Hispa/
│   └── LeafBlast/
│
├── metadata/
│   └── all_metadata.json
│
└── processed/
```

---

# 🧠 Multimodal Metadata

Mỗi ảnh được gắn:

- nhiều mô tả tiếng Việt
- semantic disease descriptions
- mô tả tổn thương lá
- symptom-level information

Ví dụ:

```json
{
  "image": "BrownSpot/img_001.jpg",
  "texts": [
    "Lá lúa xuất hiện nhiều đốm nâu nhỏ.",
    "Phiến lá có các vùng cháy màu nâu.",
    "Triệu chứng bệnh đốm nâu xuất hiện rõ."
  ],
  "label": "BrownSpot"
}
```

---

# 🔬 EDA (Exploratory Data Analysis)

## 📊 Dataset Statistics

| Class | Number of Images |
|---|---:|
| BrownSpot | 523 |
| Healthy | 1488 |
| Hispa | 565 |
| LeafBlast | 779 |

---

# 📈 Dataset Insights

## 1️⃣ Dataset Imbalance

Class Healthy có số lượng ảnh lớn hơn đáng kể so với các lớp bệnh.

### ⚠️ Potential Issues

- model bias
- over-predict Healthy
- giảm Recall của lớp bệnh
- giảm Macro F1-score

### ✅ Recommended Solutions

- Weighted Loss
- Focal Loss
- Data Augmentation
- Stratified Split
- Macro F1-score monitoring

---

## 2️⃣ Dataset Quality

### ✅ Advantages

- ảnh sạch
- object rõ ràng
- resolution cao
- triệu chứng bệnh rõ
- ít corrupted image

### ⚠️ Limitations

- thiếu ảnh thực tế ngoài đồng
- domain diversity còn hạn chế
- ánh sáng chưa đa dạng
- chưa có weather metadata thật

---

## 3️⃣ Resolution Consistency

Average resolution:

```text
2049 × 2049
```

Phù hợp cho:

- EfficientNet
- CNN
- Vision Transformer
- Transfer Learning

### Recommended Resize

- 224×224
- 256×256

---

# 📊 EDA Outputs

## 📌 Class Distribution

![Class Distribution](outputs/visualizations/class_distribution.png)

---

## 📌 Dataset Overview

![Dataset Overview](outputs/visualizations/dataset_overview.png)

---

## 📌 Image Resolution Distribution

![Resolution Distribution](outputs/visualizations/image_resolution_distribution.png)

---

## 📌 Text Length Distribution

![Text Length Distribution](outputs/visualizations/text_length_distribution.png)

---

## 📌 Texts Per Image Distribution

![Texts Per Image](outputs/visualizations/texts_per_image_distribution.png)

---

# 📋 Current EDA Features

- ✅ Class distribution analysis
- ✅ Resolution analysis
- ✅ Metadata analysis
- ✅ Text distribution analysis
- ✅ Corrupted image detection
- ✅ Automatic EDA report generation

---

# 🧪 Recommended Metrics

## Classification Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Tổng độ chính xác |
| Precision | Độ chính xác dự đoán |
| Recall | Khả năng phát hiện bệnh |
| F1-score | Cân bằng Precision & Recall |

---

# ⭐ Main Metric

## Macro F1-score

Lý do:
- dataset imbalance
- phản ánh hiệu quả giữa các class tốt hơn Accuracy

---

# 📉 Confusion Matrix

Sử dụng để:
- phân tích class dễ nhầm lẫn
- đánh giá lỗi model
- phân tích false positive / false negative

---

# 🏋️ Training Roadmap

## 🚩 Stage 1 — Image Baseline

### Model

- EfficientNet-B0

### Input

- image only

### Goal

- xây dựng baseline CNN

---

## 🚩 Stage 2 — Text Baseline

### Model

- PhoBERT

### Input

- Vietnamese descriptions

### Goal

- đánh giá semantic power của text

---

## 🚩 Stage 3 — Multimodal Fusion

### Model

- EfficientNet-B0 + PhoBERT

### Fusion

- Feature Concatenation
- Cross Attention

### Goal

- học image-text semantic alignment

---

## 🚩 Stage 4 — Advanced Multimodal Learning

### Future Improvements

- Contrastive Learning
- CLIP-style alignment
- Hard Negative Sampling
- Explainable AI
- Attention Visualization

---

# 🧠 Explainable AI Direction

## 🔥 Grad-CAM

Visualize:
- model attention
- vùng bệnh model tập trung

---

## 📝 Text Attention Visualization

Visualize:
- từ khóa quan trọng
- symptom attention

Ví dụ:
- "đốm nâu"
- "cháy lá"
- "vết bệnh"

---

# 📦 Planned Outputs

## 📊 Training Outputs

```text
outputs/
├── checkpoints/
├── logs/
├── confusion_matrix.png
├── training_curve.png
├── roc_curve.png
└── predictions/
```

---

# 🛠️ Technologies Used

## AI Frameworks

- PyTorch
- Transformers
- TorchVision
- HuggingFace

---

## NLP

- PhoBERT
- Vietnamese Tokenization

---

## Data Processing

- Pandas
- NumPy
- OpenCV
- Pillow

---

## Visualization

- Matplotlib
- Seaborn

---

# 🔬 Research Novelty

## Điểm mới của đề tài

- Vision-Language learning cho nông nghiệp
- Vietnamese agricultural NLP
- Image-text disease fusion
- Semantic disease descriptions
- Multimodal agricultural AI

---

# 🌍 Future Work

## Dataset

- thu thập ảnh thực tế ngoài đồng
- thêm weather metadata
- thêm humidity metadata
- thêm severity labels
- thêm geolocation metadata

---

## AI Model

- Vision-Language Transformer
- Contrastive Learning
- Explainable AI
- Lightweight inference
- Mobile AI deployment

---

## System

- Smart farming assistant
- Agricultural chatbot
- Real-time disease detection
- Edge AI deployment

---

# 🎓 Research Direction

Đề tài hướng tới:

- Multimodal AI
- Agricultural AI
- Vietnamese Vision-Language Models
- Smart Farming Systems
- AI for Precision Agriculture

---

# 📌 Current Project Status

## Completed

- ✅ Dataset collection
- ✅ Metadata generation
- ✅ EDA analysis
- ✅ Visualization pipeline
- ✅ Multimodal dataset structure

---

## In Progress

- 🔄 Train/Validation/Test split
- 🔄 Baseline model training
- 🔄 Multimodal fusion training

---

## Planned

- ⏳ Explainable AI
- ⏳ Contrastive learning
- ⏳ Real-world deployment
- ⏳ Mobile inference

---

# ✨ Conclusion

Dự án hướng tới xây dựng một hệ thống AI đa phương thức
cho nhận dạng sâu bệnh cây lúa bằng tiếng Việt.

Việc kết hợp:
- ảnh lá lúa
- semantic disease descriptions
- multimodal learning

mở ra hướng tiếp cận mới cho:
- Smart Farming
- Agricultural AI
- Vietnamese Vision-Language Research
- Precision Agriculture
