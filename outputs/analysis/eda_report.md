# Báo cáo EDA bộ dữ liệu bệnh lá lúa

## 1. Tổng quan bộ dữ liệu

- Tổng số ảnh: **3.355**
- Số lớp: **4**
  - `Healthy`: 1.488 ảnh
  - `LeafBlast`: 779 ảnh
  - `Hispa`: 565 ảnh
  - `BrownSpot`: 523 ảnh

## 2. Thống kê metadata chính

- Trung bình mỗi ảnh có **5 câu mô tả** (`texts`).
- Độ dài văn bản trung bình: **118 từ/ảnh**.
- Số mẫu văn bản độc nhất: **3.355**.
- Phân bố `metadata_quality`:
  - `high`: 1.239
  - `medium`: 1.464
  - `low`: 652
- Phân bố `severity`:
  - `none`: 1.488
  - `moderate`: 941
  - `mild`: 525
  - `severe`: 401

## 3. Phân tích ảnh

### 3.1 Phân bố lớp

Bộ dữ liệu mất cân bằng rõ rệt:
- `Healthy`: 44,4%
- `LeafBlast`: 23,2%
- `Hispa`: 16,8%
- `BrownSpot`: 15,6%

### 3.2 Đặc điểm hình ảnh

- `outputs/visualizations/class_visual_collage.png` cho thấy ảnh chủ yếu là close-up trên nền trắng.
- Ảnh có độ phân giải cao, ít nhiễu nhưng thiếu đa dạng bối cảnh thực địa.
- `lesion_area_ratio` trung bình khoảng **0,015**, nghĩa là vùng bệnh thường nhỏ và tinh vi.

## 4. Phân tích văn bản

### 4.1 Độ dài và cấu trúc

- Độ dài văn bản nằm trong khoảng **59–536 từ/ảnh**.
- Phần lớn văn bản tập trung quanh **100–120 từ**.
- `outputs/visualizations/text_length_distribution.png` xác nhận phần lớn mô tả có độ dài vừa phải.

### 4.2 Đa dạng nội dung

- Mỗi ảnh có trung bình 5 câu mô tả.
- Văn bản chứa thông tin triệu chứng và phân tích trực quan.
- Nhiều đoạn mô tả vẫn tuân theo khuôn mẫu chuyên môn, gây rủi ro học theo mẫu ngôn ngữ.

## 5. Rủi ro dữ liệu

- Mất cân bằng lớp: `Healthy` chiếm tỷ trọng lớn.
- Thiếu đa dạng nền ảnh: phần lớn ảnh nền trắng.
- Tổn thương nhỏ và lesion ratio thấp.
- Metadata chất lượng thấp chiếm gần 20% tổng dữ liệu.

## 6. Khả năng đa phương thức

- Dữ liệu phù hợp cho học đa phương thức nhờ ảnh và văn bản liên kết.
- Trường thông tin thời tiết và giai đoạn sinh trưởng hỗ trợ phân tích điều kiện môi trường.
- Tuy nhiên, dạng ảnh đồng nhất và văn bản khuôn mẫu hạn chế tính khái quát.

## 7. Quan sát từ visualizations

- `class_distribution.png` khẳng định mất cân bằng lớp.
- `class_visual_collage.png` xác nhận ảnh close-up nền trắng.
- `text_length_distribution.png` cho thấy độ dài văn bản phù hợp.
- `image_text_pairs.png` minh chứng dữ liệu ảnh–văn bản liên kết.

## 8. Kết luận

Bộ dữ liệu phù hợp cho nghiên cứu phân loại bệnh lá lúa và thử nghiệm học đa phương thức tiếng Việt. Tuy nhiên, để đạt chất lượng báo cáo luận văn hoặc đề tài tốt nghiệp, cần:
- bổ sung ảnh thực địa nền phức hợp,
- xử lý mất cân bằng lớp,
- ưu tiên đánh giá trên subset chất lượng cao,
- thiết kế thêm thử nghiệm đối chiếu khi sử dụng metadata bệnh lý.
