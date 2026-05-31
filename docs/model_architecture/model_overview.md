# MODEL ARCHITECTURE

## Final Multimodal Architecture

### Vision Branch

Input:

Rice Leaf Image

Preprocessing:

* Resize 224×224
* Normalize

Encoder:

EfficientNet-B0

Output:

Image Embedding (768-d)

---

### Text Branch

Input:

Vietnamese Description

Preprocessing:

* Tokenization
* Attention Mask

Encoder:

PhoBERT

Output:

Text Embedding (768-d)

---

### Multimodal Fusion

Primary Fusion Strategy:

Cross-Attention

Image Feature
↔
Text Feature

Cross-modal interaction giúp mô hình học được mối liên hệ giữa:

* triệu chứng mô tả bằng văn bản
* đặc trưng thị giác trên lá lúa

---

### Baseline Fusion

Feature Concatenation

Image Embedding
+
Text Embedding

↓

MLP

---

### Classification Head

MLP Classifier

↓

Softmax

↓

4-Class Prediction

* BrownSpot
* Healthy
* Hispa
* LeafBlast

---

## Why EfficientNet-B0?

* nhẹ
* pretrained ImageNet
* phù hợp dataset kích thước vừa
* dễ fine-tune

---

## Why PhoBERT?

* pretrained trên tiếng Việt
* hiệu quả cao cho downstream tasks
* phù hợp metadata tiếng Việt

---

## Why Cross-Attention?

Concat chỉ ghép đặc trưng.

Cross-Attention cho phép:

* Image → attend Text
* Text → attend Image

giúp khai thác thông tin đa phương thức hiệu quả hơn.

---

## Expected Research Contribution

* Vietnamese agricultural multimodal dataset
* Cross-modal disease recognition
* Explainable AI for rice disease diagnosis
* Vision-Language Learning in agriculture
