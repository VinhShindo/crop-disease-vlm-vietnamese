# Rice Leaf Disease — Dữ liệu mô hình đa phương thức

## 1. Mục tiêu dự án

Mục tiêu chính của dự án là xây dựng và phân tích bộ dữ liệu ảnh–văn bản tiếng Việt cho bệnh lá lúa, phục vụ nghiên cứu Vision-Language Model trong nông nghiệp. Báo cáo này đánh giá chất lượng dữ liệu, mức độ đa dạng và rủi ro khi sử dụng bộ dữ liệu cho tác vụ phân loại bệnh và học đa phương thức.

## 2. Tổng quan dataset

- Tổng số mẫu ảnh: **3.355**
- Số lớp: **4**
  - `Healthy` (Lá khỏe): **1.488 ảnh** (44,4%)
  - `LeafBlast` (Đạo ôn lá): **779 ảnh** (23,2%)
  - `Hispa` (Sâu hispa): **565 ảnh** (16,8%)
  - `BrownSpot` (Đốm nâu): **523 ảnh** (15,6%)
- Phân bố tại 6 địa phương: `Can Tho`, `An Giang`, `Long An`, `Dong Thap`, `Tien Giang`, `Thai Binh`
- Giai đoạn sinh trưởng: `tillering`, `seedling`, `booting`, `flowering`, `ripening`
- Thời tiết: `cloudy`, `sunny`, `rainy`
- Chất lượng metadata: `high` 1.239, `medium` 1.464, `low` 652

## 3. Cấu trúc metadata

Bộ dữ liệu chứa các trường chính sau:
- `image`
- `label`, `vietnamese_label`
- `texts`, `symptoms`, `visual_analysis`
- `leaf_area_ratio`, `background_ratio`, `lesion_area_ratio`
- `annotation_confidence`, `metadata_quality`
- `weather`, `humidity`, `temperature`, `severity`
- `growth_stage`, `location`, `farmer_note`

## 4. Điểm mạnh của dữ liệu

- Văn bản mô tả tiếng Việt chi tiết, phù hợp cho mô hình đa phương thức.
- Mỗi ảnh đi kèm trung bình **5 câu** mô tả và khoảng **118 từ**.
- Toàn bộ **3.355 mẫu văn bản đều độc nhất**, nghĩa là không có text trùng lặp toàn bộ giữa các ảnh.
- Metadata bổ sung (thời tiết, giai đoạn, độ tin cậy) là tài nguyên quan trọng cho phân tích điều kiện thực địa.

## 5. Quan sát từ hình ảnh

- `outputs/visualizations/class_visual_collage.png` cho thấy ảnh chủ yếu là close-up trên nền trắng, thiếu bối cảnh đồng ruộng.
- Ảnh có độ phân giải cao, đồng đều và ít nhiễu, nhưng điều này giảm độ đa dạng miền ảnh thực địa.
- `lesion_area_ratio` trung bình khoảng **0,015**, nghĩa là vùng bệnh thường nhỏ và tinh vi.

## 6. Rủi ro và hạn chế

- Mất cân bằng lớp: `Healthy` chiếm 44,4%.
- Thiếu đa dạng nền ảnh: đa số ảnh nền trắng, hạn chế khả năng khái quát.
- Kích thước tổn thương nhỏ gây độ khó cho phân loại.
- Metadata chất lượng thấp chiếm gần 20% tổng dữ liệu.

## 7. Khả năng đa phương thức

- Bộ dữ liệu có nền tảng tốt cho học đa phương thức vì ảnh và văn bản được liên kết trực tiếp.
- Trường thông tin thời tiết và giai đoạn sinh trưởng cho phép mở rộng phân tích theo điều kiện môi trường.
- Tuy nhiên, dạng ảnh đồng nhất và khuôn mẫu văn bản còn hạn chế tính khái quát.

## 8. Ứng dụng đề xuất

- Phân loại bệnh lá lúa bốn nhãn cơ bản.
- Nghiên cứu VLM tiếng Việt cho nông nghiệp.
- Hệ thống hỗ trợ chẩn đoán bệnh lúa dựa trên ảnh và metadata.

## 9. Hướng dẫn nhanh

1. Kích hoạt môi trường:
```bash
source rice_plant_venv/bin/activate
```
2. Chạy EDA:
```bash
./rice_plant_venv/bin/python src/datasets/EDA.py
```
3. Kiểm tra kết quả output:
- `outputs/visualizations/`
- `outputs/analysis/eda_report.md`

## 10. Khuyến nghị

- Sử dụng chiến lược cân bằng lớp hoặc augmentation cho `BrownSpot` và `Hispa`.
- Bổ sung dữ liệu thực địa có nền phong phú.
- Giữ subset `high` metadata quality làm đối chứng.
- Kết hợp loss phân lớp và loss contrastive để khai thác cả ảnh và văn bản.
