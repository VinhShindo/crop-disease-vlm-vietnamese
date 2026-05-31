# EVALUATION PROTOCOL

## Classification Metrics

### Accuracy

Tỷ lệ dự đoán đúng.

---

### Precision

Độ chính xác của lớp dự đoán.

---

### Recall

Khả năng phát hiện đầy đủ bệnh.

---

### Macro F1-score

Metric chính của nghiên cứu.

Lý do:

Dataset bị mất cân bằng lớp.

Macro F1 phản ánh tốt hơn Accuracy.

---

## Confusion Matrix

Được sử dụng để xác định:

* lớp dễ nhầm lẫn
* lớp khó phân biệt

Ví dụ:

BrownSpot ↔ LeafBlast

Healthy ↔ LeafBlast

---

## Explainability Metrics

### Grad-CAM

Xác định vùng ảnh ảnh hưởng đến dự đoán.

### Attention Visualization

Phân tích tương tác:

Image Tokens
↔
Text Tokens

---

## Embedding Analysis

### t-SNE

Giảm chiều embedding.

Đánh giá:

* cluster quality
* class separation

### UMAP

Đánh giá cấu trúc embedding ở không gian thấp chiều.

---

## Future Evaluation

### Retrieval Task

Image → Text

Text → Image

Metrics:

* Recall@1
* Recall@5
* Recall@10

---

## Expected Outputs

outputs/

├── checkpoints/
├── metrics/
├── confusion_matrix/
├── embeddings/
├── gradcam/
├── attention_maps/
├── predictions/
└── reports/

Các artifact này được sử dụng trong luận văn, báo cáo nghiên cứu và bài báo khoa học.
