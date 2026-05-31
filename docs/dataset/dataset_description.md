# DATASET DOCUMENTATION

## Dataset Overview

Tên bộ dữ liệu:

Rice Leaf Disease Multimodal Dataset (Vietnamese)

Mục tiêu:

Phân loại bệnh lá lúa bằng dữ liệu đa phương thức (Multimodal Learning), kết hợp:

* Ảnh lá lúa
* Mô tả tiếng Việt
* Metadata nông nghiệp

---

## Dataset Statistics

Tổng số ảnh:

3355

Số lớp:

4

| Class     | Samples |
| --------- | ------: |
| Healthy   |    1488 |
| LeafBlast |     779 |
| Hispa     |     565 |
| BrownSpot |     523 |

---

## Disease Classes

### Healthy

Lá lúa khỏe mạnh không xuất hiện tổn thương.

### BrownSpot

Bệnh đốm nâu.

Triệu chứng:

* đốm nâu nhỏ
* cháy lá cục bộ
* mô lá khô

### LeafBlast

Bệnh đạo ôn lá.

Triệu chứng:

* vết hình thoi
* tâm xám
* mép nâu

### Hispa

Bệnh sâu hispa.

Triệu chứng:

* vệt trắng bạc
* lá bị ăn biểu bì
* mô lá mất sắc tố

---

## Metadata Structure

Mỗi mẫu dữ liệu bao gồm:

* image
* label
* vietnamese_label

### Textual Description

* texts
* symptoms
* visual_analysis

### Agricultural Metadata

* weather
* humidity
* temperature
* severity
* growth_stage
* location
* farmer_note

### Dataset Quality Metadata

* annotation_confidence
* metadata_quality

### Visual Metadata

* leaf_area_ratio
* lesion_area_ratio
* background_ratio

---

## Example Sample

Một bản ghi dữ liệu bao gồm:

Ảnh lá lúa
↓
Mô tả triệu chứng
↓
Thông tin môi trường
↓
Nhãn bệnh

---

## Dataset Challenges

### Class Imbalance

Healthy chiếm khoảng 44% dữ liệu.

### Metadata Diversity

Mô tả văn bản được sinh tự động nên còn hạn chế về đa dạng ngữ nghĩa.

### Domain Gap

Phần lớn ảnh được chụp nền trắng hoặc nền đơn giản.

Chưa phản ánh đầy đủ điều kiện đồng ruộng thực tế.

---

## Recommended Future Collection

Nên bổ sung:

* ảnh ngoài thực địa
* nhiều điều kiện ánh sáng
* nhiều giống lúa
* nhiều vùng địa lý
* nhiều mức độ bệnh

Để tăng khả năng tổng quát hóa của mô hình.
