from pathlib import Path
import json
import random

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = Path("dataset/raw")

OUTPUT_DIR = Path("dataset/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "all_metadata.json"

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =========================================================
# NUMBER OF TEXTS PER IMAGE
# =========================================================

MIN_TEXTS_PER_IMAGE = 3
MAX_TEXTS_PER_IMAGE = 5

# =========================================================
# DETAILED VIETNAMESE DISEASE DESCRIPTIONS
# =========================================================

TEXT_TEMPLATES = {

    # =====================================================
    # BROWN SPOT
    # =====================================================

    "BrownSpot": [

        "Lá lúa xuất hiện nhiều đốm nâu nhỏ rải rác trên bề mặt lá.",
        "Các vết bệnh màu nâu đậm xuất hiện trên phiến lá lúa.",
        "Lá có các đốm nâu hình oval với viền vàng nhạt.",
        "Xuất hiện nhiều chấm nâu nhỏ lan rộng trên lá.",
        "Bề mặt lá xuất hiện các tổn thương màu nâu sẫm.",
        "Lá bị cháy nhẹ với nhiều đốm nâu phân bố không đều.",
        "Các đốm bệnh màu nâu bắt đầu lan rộng trên phiến lá.",
        "Lá xuất hiện các vùng hoại tử màu nâu.",
        "Vết bệnh hình bầu dục màu nâu xuất hiện rõ trên lá.",
        "Các tổn thương màu nâu làm lá có dấu hiệu khô cháy.",
        "Lá lúa xuất hiện nhiều vết nâu nhỏ do bệnh đốm nâu.",
        "Triệu chứng bệnh đốm nâu xuất hiện dày đặc trên lá.",
        "Phiến lá bị tổn thương bởi các vết nâu đậm.",
        "Lá có dấu hiệu khô mép và xuất hiện các đốm nâu.",
        "Nhiều vùng màu nâu xuất hiện trên bề mặt lá lúa.",
        "Lá lúa bị bệnh với các đốm nâu tròn nhỏ.",
        "Các đốm nâu liên kết thành vùng lớn trên lá.",
        "Triệu chứng bệnh gây cháy lá và xuất hiện đốm nâu.",
        "Lá xuất hiện các tổn thương màu nâu nhạt đến nâu đậm.",
        "Đốm nâu xuất hiện nhiều ở phần giữa của phiến lá.",
        "Lá lúa có nhiều đốm nâu nhỏ với phần trung tâm màu xám nhạt.",
        "Các vết nâu xuất hiện tập trung ở phần đầu lá.",
        "Triệu chứng bệnh đốm nâu làm giảm màu xanh tự nhiên của lá.",
        "Phiến lá có nhiều vùng tổn thương màu nâu lan rộng.",
        "Lá bị khô cục bộ do các vết bệnh màu nâu.",
        "Đốm bệnh có dạng hình bầu dục và phân bố rải rác trên lá.",
        "Mặt lá xuất hiện nhiều vết hoại tử màu nâu đậm.",
        "Triệu chứng bệnh làm lá xuất hiện nhiều vùng cháy nâu.",
        "Lá lúa có dấu hiệu héo nhẹ kèm các đốm nâu nhỏ.",
        "Các đốm bệnh màu nâu làm giảm diện tích quang hợp của lá."
    ],

    # =====================================================
    # HEALTHY
    # =====================================================

    "Healthy": [

        "Lá lúa khỏe mạnh với màu xanh đồng đều.",
        "Không phát hiện dấu hiệu bệnh trên lá lúa.",
        "Phiến lá phát triển bình thường và không có tổn thương.",
        "Lá có màu xanh tự nhiên và bề mặt sạch.",
        "Lá lúa sinh trưởng tốt và không có dấu hiệu sâu bệnh.",
        "Bề mặt lá nguyên vẹn và không xuất hiện đốm bệnh.",
        "Lá phát triển khỏe mạnh trong điều kiện bình thường.",
        "Không phát hiện vết cháy hoặc vùng tổn thương trên lá.",
        "Lá có màu xanh sáng và phát triển đồng đều.",
        "Lá không có dấu hiệu nhiễm nấm hoặc sâu bệnh.",
        "Phiến lá xanh tốt và không bị khô mép.",
        "Lá phát triển bình thường với màu sắc ổn định.",
        "Không xuất hiện triệu chứng bất thường trên lá.",
        "Lá giữ được trạng thái khỏe mạnh và cân đối.",
        "Bề mặt lá mịn và không có tổn thương cơ học.",
        "Lá lúa có khả năng sinh trưởng tốt.",
        "Không có dấu hiệu vàng lá hoặc đốm bệnh.",
        "Lá phát triển mạnh với màu xanh tự nhiên.",
        "Phiến lá hoàn toàn khỏe mạnh và không có vết bệnh.",
        "Lá không có dấu hiệu bị sâu ăn hoặc cháy lá.",
        "Lá lúa xanh đều từ gốc đến ngọn và không có triệu chứng bất thường.",
        "Cây lúa phát triển khỏe mạnh với bộ lá xanh tốt.",
        "Lá có độ bóng tự nhiên và không bị tổn thương.",
        "Không phát hiện các vùng hoại tử hoặc cháy lá.",
        "Màu sắc lá đồng nhất và không xuất hiện đốm lạ.",
        "Lá lúa phát triển ổn định trong điều kiện bình thường.",
        "Bề mặt lá sạch và không có dấu hiệu nhiễm bệnh.",
        "Phiến lá duy trì trạng thái khỏe mạnh và xanh tốt.",
        "Lá không có dấu hiệu mất màu hoặc biến dạng.",
        "Triệu chứng sinh trưởng bình thường được quan sát rõ trên lá."
    ],

    # =====================================================
    # HISPA
    # =====================================================

    "Hispa": [

        "Lá lúa xuất hiện nhiều vệt trắng do sâu ăn phá.",
        "Phiến lá bị sâu Hispa gây hại tạo thành các sọc trắng.",
        "Bề mặt lá xuất hiện các đường trắng kéo dài.",
        "Lá bị côn trùng ăn làm mất lớp biểu bì.",
        "Xuất hiện nhiều vùng bạc màu trên phiến lá.",
        "Lá có dấu hiệu bị sâu phá hoại nghiêm trọng.",
        "Các sọc trắng xuất hiện dọc theo chiều dài lá.",
        "Sâu Hispa gây tổn thương làm lá bị khô nhẹ.",
        "Phiến lá xuất hiện nhiều vết cào xước màu trắng.",
        "Lá bị sâu ăn làm giảm màu xanh tự nhiên.",
        "Các vùng trắng bạc xuất hiện rõ trên bề mặt lá.",
        "Lá bị tổn thương bởi hoạt động ăn lá của sâu.",
        "Nhiều đường trắng song song xuất hiện trên lá.",
        "Bề mặt lá bị mất mô xanh do sâu gây hại.",
        "Triệu chứng sâu Hispa làm lá giảm khả năng quang hợp.",
        "Lá có nhiều vết ăn kéo dài theo phiến lá.",
        "Các vùng tổn thương trắng xuất hiện tập trung trên lá.",
        "Sâu gây hại làm lá có dấu hiệu khô và bạc màu.",
        "Phiến lá bị ăn mòn tạo thành các sọc dài.",
        "Lá lúa bị sâu Hispa gây ra nhiều tổn thương cơ học.",
        "Lá bị sâu phá hại tạo thành các dải trắng dài trên phiến lá.",
        "Các vùng bạc màu xuất hiện do sâu ăn lớp mô xanh.",
        "Triệu chứng sâu hại làm lá mất màu xanh tự nhiên.",
        "Bề mặt lá xuất hiện nhiều vết tổn thương dạng sọc.",
        "Lá bị hư hại nghiêm trọng bởi côn trùng gây hại.",
        "Các đường trắng phân bố dọc theo phiến lá lúa.",
        "Lá có dấu hiệu bị cào xước và bạc màu rõ rệt.",
        "Sâu Hispa làm giảm chất lượng và diện tích lá xanh.",
        "Các tổn thương cơ học xuất hiện trên toàn bộ phiến lá.",
        "Lá bị sâu gây hại với nhiều vùng mất mô xanh."
    ],

    # =====================================================
    # LEAF BLAST
    # =====================================================

    "LeafBlast": [

        "Lá lúa xuất hiện các vết cháy hình thoi màu nâu xám.",
        "Triệu chứng bệnh đạo ôn xuất hiện rõ trên phiến lá.",
        "Các đốm cháy màu xám xuất hiện trên lá lúa.",
        "Lá bị bệnh blast với nhiều vết tổn thương kéo dài.",
        "Vết bệnh hình thoi xuất hiện dọc theo phiến lá.",
        "Các vùng cháy màu nâu xám lan rộng trên lá.",
        "Lá có dấu hiệu khô cháy do bệnh đạo ôn.",
        "Triệu chứng bệnh làm lá xuất hiện nhiều đốm cháy.",
        "Phiến lá xuất hiện tổn thương màu nâu ở trung tâm.",
        "Các vết bệnh hình oval kéo dài xuất hiện trên lá.",
        "Lá bị cháy cục bộ với nhiều vùng hoại tử.",
        "Bệnh đạo ôn gây ra các tổn thương nghiêm trọng trên lá.",
        "Vết cháy màu xám có viền nâu xuất hiện rõ.",
        "Lá xuất hiện nhiều đốm bệnh liên kết thành mảng lớn.",
        "Các tổn thương lan rộng làm lá bị khô.",
        "Phiến lá có nhiều vết cháy hình thoi đặc trưng.",
        "Lá bị hoại tử do nấm gây bệnh blast.",
        "Các vùng bệnh màu nâu xám xuất hiện tập trung.",
        "Triệu chứng cháy lá xuất hiện trên nhiều vị trí.",
        "Bệnh blast làm lá mất khả năng quang hợp.",
        "Lá lúa có nhiều tổn thương hình thoi màu xám nhạt.",
        "Các vết cháy xuất hiện tập trung ở phần giữa phiến lá.",
        "Triệu chứng bệnh đạo ôn làm lá bị khô nhanh.",
        "Vết bệnh có trung tâm màu xám và viền nâu đậm.",
        "Lá xuất hiện các vùng hoại tử kéo dài theo phiến lá.",
        "Bệnh blast gây cháy lá và làm lá suy giảm sinh trưởng.",
        "Các đốm cháy lan rộng tạo thành mảng lớn trên lá.",
        "Phiến lá bị tổn thương nặng với nhiều vết cháy đặc trưng.",
        "Lá bị nấm bệnh gây ra các vùng cháy màu xám nâu.",
        "Triệu chứng bệnh làm lá khô và biến màu nghiêm trọng."
    ]
}

# =========================================================
# AUTO DETECT CLASSES
# =========================================================

CLASSES = sorted([
    folder.name
    for folder in DATASET_ROOT.iterdir()
    if folder.is_dir()
])

print("=" * 60)
print("GENERATING MULTI-TEXT VIETNAMESE METADATA")
print("=" * 60)

print("\nDetected classes:")

for cls in CLASSES:
    print(f"- {cls}")

# =========================================================
# GENERATE METADATA
# =========================================================

metadata = []

total_images = 0

for cls in CLASSES:

    class_dir = DATASET_ROOT / cls

    image_files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.jpeg"))
    )

    print(f"\nProcessing class: {cls}")
    print(f"Images found: {len(image_files)}")

    templates = TEXT_TEMPLATES.get(cls, [])

    for img_path in image_files:

        relative_path = img_path.relative_to(DATASET_ROOT)

        # =================================================
        # RANDOM NUMBER OF TEXTS
        # =================================================

        num_texts = random.randint(
            MIN_TEXTS_PER_IMAGE,
            MAX_TEXTS_PER_IMAGE
        )

        # =================================================
        # RANDOM UNIQUE DESCRIPTIONS
        # =================================================

        descriptions = random.sample(
            templates,
            k=min(num_texts, len(templates))
        )

        sample = {

            "image": str(relative_path).replace("\\", "/"),

            "texts": descriptions,

            "label": cls,

            "language": "vi",

            "num_texts": len(descriptions),

            "modality": [
                "image",
                "text"
            ],

            "task": "rice_leaf_disease_classification"
        }

        metadata.append(sample)

        total_images += 1

# =========================================================
# SAVE JSON
# =========================================================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=4
    )

# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("=" * 60)
print("METADATA GENERATION COMPLETED")
print("=" * 60)

print(f"\nTotal samples: {total_images}")

print(f"\nSaved JSON:")
print(OUTPUT_JSON)

print("\nExample sample:\n")

print(json.dumps(
    metadata[0],
    ensure_ascii=False,
    indent=4
))