# SLIDE_REPORT

## Slide 1 — Tiêu đề & Mục tiêu
- Nội dung:
  - Tên đề tài: "Phân tích bộ dữ liệu ảnh-văn bản tiếng Việt cho bệnh lá lúa"
  - Mục tiêu nghiên cứu: đánh giá dataset EDA, xác định rủi ro và tiềm năng multimodal.
- Hình ảnh:
  - Không cần hình chính, chỉ slide tiêu đề.
- Insight:
  - Nhấn mạnh dataset là cơ sở cho VLM nông nghiệp tiếng Việt.
- Lời thuyết trình:
  - "Mục tiêu của chúng tôi là thực chứng chất lượng và tính khả thi của dữ liệu ảnh-văn bản cho bệnh lá lúa bằng cách phân tích cả ảnh và metadata."

## Slide 2 — Thống kê cơ bản dataset
- Nội dung:
  - Tổng số ảnh: 3.355
  - Số lớp: 4 (Healthy, LeafBlast, Hispa, BrownSpot)
  - Phân bố địa phương và giai đoạn sinh trưởng.
- Hình ảnh:
  - `outputs/visualizations/dataset_overview.png`
- Insight:
  - Dataset đủ lớn cho phân loại đa lớp nhưng cần quan tâm mất cân bằng.
- Lời thuyết trình:
  - "Bộ dữ liệu hiện có 3.355 ảnh, thu thập từ 6 tỉnh, trải dài nhiều giai đoạn sinh trưởng. Đây là nền tảng tốt nhưng vẫn cần xử lý imbalance."

## Slide 3 — Mất cân bằng lớp
- Nội dung:
  - `Healthy` 44,4%; `LeafBlast` 23,2%; `Hispa` 16,8%; `BrownSpot` 15,6%.
- Hình ảnh:
  - `outputs/visualizations/class_distribution.png`
- Insight:
  - Cần chiến lược augmentation hoặc weighting cho lớp bệnh.
- Lời thuyết trình:
  - "Phân bố lớp không đồng đều: lớp Healthy chiếm gần một nửa, điều này yêu cầu kỹ thuật đào tạo để tránh mô hình thiên vị."

## Slide 4 — Đặc điểm hình ảnh
- Nội dung:
  - Ảnh chủ yếu close-up trên nền trắng.
  - Vùng bệnh chiếm diện tích nhỏ.
- Hình ảnh:
  - `outputs/visualizations/class_visual_collage.png`
- Insight:
  - Dữ liệu có chất lượng ảnh tốt nhưng thiếu đa dạng hoàn cảnh.
- Lời thuyết trình:
  - "Ảnh tập trung vào phiến lá trên nền trắng. Điều này tốt cho việc học đặc trưng bệnh nhưng hạn chế khả năng mở rộng ra ảnh hiện trường."

## Slide 5 — Metadata văn bản
- Nội dung:
  - Trung bình 5 câu/ảnh, 118 từ/ảnh.
  - 3.355 mẫu văn bản độc nhất.
- Hình ảnh:
  - `outputs/visualizations/text_length_distribution.png`
  - `outputs/visualizations/image_text_pairs.png`
- Insight:
  - Văn bản phong phú, phù hợp cho multimodal, nhưng cần kiểm soát template.
- Lời thuyết trình:
  - "Mỗi ảnh kèm mô tả tiếng Việt dài hơn 100 từ, hữu ích cho học kết hợp ảnh và ngôn ngữ. Tuy nhiên, nội dung vẫn có cấu trúc lặp cần theo dõi."

## Slide 6 — Rủi ro dữ liệu
- Nội dung:
  - Mất cân bằng lớp.
  - Ảnh nền trắng, ít bối cảnh đồng ruộng.
  - Metadata chất lượng thấp chiếm gần 20%.
- Hình ảnh:
  - `outputs/visualizations/class_distribution.png`
  - `outputs/visualizations/text_length_distribution.png`
- Insight:
  - Cần lọc dữ liệu và tăng cường đa dạng.
- Lời thuyết trình:
  - "Các rủi ro chính là imbalance, thiếu đa dạng miền ảnh và metadata chất lượng thấp. Đây là điểm cần cải thiện trước khi đưa mô hình vào thực tế."

## Slide 7 — Khả năng multimodal
- Nội dung:
  - Dữ liệu đủ điều kiện cho học ảnh–văn bản.
  - Metadata bổ sung như điều kiện thời tiết và giai đoạn.
- Hình ảnh:
  - `outputs/visualizations/image_text_pairs.png`
- Insight:
  - Dữ liệu có tiềm năng VLM nhưng cần kiểm soát generalization.
- Lời thuyết trình:
  - "Dữ liệu hiện tại có ngữ nghĩa đa phương thức tốt, đặc biệt với thông tin điều kiện trồng trọt. Tuy vậy, chúng ta cần thử nghiệm thêm với ảnh hiện trường để đánh giá khả năng khái quát."

## Slide 8 — Khuyến nghị nghiên cứu
- Nội dung:
  - Augmentation lớp bệnh.
  - Bổ sung ảnh nền thực địa.
  - Dùng subset quality cao để validate.
  - Thử nghiệm loss contrastive và class-weighted loss.
- Hình ảnh:
  - Không cần hình chính, có thể dùng icon hoặc sơ đồ logic.
- Insight:
  - Hướng tiếp theo là làm sạch và mở rộng dữ liệu.
- Lời thuyết trình:
  - "Để nâng chất lượng nghiên cứu, chúng tôi đề xuất cân bằng lớp, bổ sung ảnh thực địa, và sử dụng dữ liệu chất lượng cao làm tiêu chuẩn đánh giá."

## Slide 9 — Kết luận
- Nội dung:
  - Dataset phù hợp cho nghiên cứu phân loại bệnh lá lúa và VLM.
  - Cần xử lý imbalance và mở rộng đa dạng ảnh.
  - Metadata tiếng Việt là điểm mạnh.
- Hình ảnh:
  - `outputs/visualizations/dataset_overview.png`
- Insight:
  - Bộ dữ liệu là nền tảng, không phải sản phẩm cuối cùng.
- Lời thuyết trình:
  - "Bộ dữ liệu là cơ sở hữu ích cho luận văn, nhưng để đạt được kết quả ổn định, cần tiếp tục làm giàu và hiệu chỉnh thêm."
