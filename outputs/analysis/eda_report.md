# FULL EDA REPORT

# 1. Dataset Overview

Dataset hiện tại sử dụng:
- Rice Leaf Diseases Dataset

Bao gồm 4 classes:
- BrownSpot
- Healthy
- Hispa
- LeafBlast

---

# 2. Dataset Statistics

| Class | Images |
|---|---|
| BrownSpot | 523 |
| Healthy | 1488 |
| Hispa | 565 |
| LeafBlast | 779 |

Tổng số classes:
- 4

---

# 3. Dataset Quality Assessment

## Ưu điểm

- Ảnh chất lượng tốt
- Ít corrupted image
- Background đơn giản
- Leaf region rõ ràng
- Ít nhiễu
- Resolution tương đối đồng đều

## Nhược điểm

- Chưa phản ánh điều kiện thực tế ngoài đồng
- Background quá sạch
- Thiếu ánh sáng tự nhiên
- Thiếu shadow/noise
- Thiếu ảnh nhiều lá

---

# 4. Dataset Imbalance Analysis

Class Healthy có số lượng ảnh lớn nhất.

Điều này gây nguy cơ:
- Model bias
- Overfitting vào lớp Healthy
- Accuracy cao giả tạo

## Đề xuất xử lý

- Weighted Cross Entropy
- Data Augmentation
- F1-score monitoring
- Stratified splitting

---

# 5. Resolution Analysis

Resolution ảnh khá đồng đều.

Điều này giúp:
- training ổn định
- preprocessing đơn giản
- batching hiệu quả

Recommended input size:
- 224x224
- 256x256

---

# 6. Model Suitability Analysis

Dataset hiện tại phù hợp với:

## Highly Recommended
- EfficientNet-B0
- ResNet50
- ConvNeXt

## Acceptable
- Vision Transformer (ViT)

## Not Recommended
- Video models
- Large multimodal transformers

Lý do:
Dataset là ảnh tĩnh, không có temporal information.

---

# 7. Implication for Vision-Language Learning

Dataset hiện tại mới chỉ có image modality.

Cần bổ sung:
- mô tả tiếng Việt
- metadata thời tiết
- độ ẩm
- vị trí địa lý

Ví dụ:

```json
{
    "image": "LeafBlast_001.jpg",
    "text": "Lá lúa xuất hiện vết cháy màu nâu vàng.",
    "weather": "rainy",
    "humidity": 85
}
````

---

# 8. Recommended Pipeline

## Step 1

EDA & dataset validation

## Step 2

Metadata generation

## Step 3

Image preprocessing

## Step 4

Train EfficientNet baseline

## Step 5

Train multimodal model:

* EfficientNet
* PhoBERT
* Fusion Layer

## Step 6

Evaluation

---

# 9. Evaluation Metrics

## Classification

* Accuracy
* Precision
* Recall
* F1-score

Main metric:

* F1-score

---

# 10. Conclusions

Dataset hiện tại:

* phù hợp cho baseline AI model
* phù hợp cho luận văn
* phù hợp cho multimodal research

Tuy nhiên:

* cần thêm dữ liệu thực tế
* cần metadata tiếng Việt
* cần augmentation mạnh để tăng generalization
