"""Generate image-grounded Vietnamese metadata for rice leaf disease dataset.

This script reads dataset/raw/<class> images and produces:
- dataset/metadata/all_metadata.json
- dataset/metadata/metadata.csv
- dataset/metadata/metadata_summary.json

The metadata includes field values derived from image pixel analysis and
class-specific description templates to improve visual-text alignment.
"""

from pathlib import Path
import json
import random
from collections import Counter

import numpy as np
from PIL import Image
import pandas as pd

DATASET_ROOT = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "all_metadata.json"
OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "metadata_summary.json"

random.seed(42)

CLASS_INFO = {
    "BrownSpot": {
        "vietnamese_label": "Đốm nâu",
        "symptoms": [
            "đốm nâu nhỏ trên lá",
            "vết cháy màu nâu",
            "lá khô từng vùng",
            "mép lá chuyển nâu",
            "phiến lá xuất hiện đốm tròn"
        ],
        "templates": [
            "Lá lúa xuất hiện nhiều đốm nâu nhỏ rải rác trên bề mặt lá.",
            "Các vết bệnh màu nâu đậm xuất hiện trên phiến lá lúa.",
            "Lá có các đốm nâu hình oval với viền vàng nhạt.",
            "Xuất hiện nhiều chấm nâu nhỏ lan rộng trên lá.",
            "Bề mặt lá xuất hiện các tổn thương màu nâu sẫm.",
            "Lá bị cháy nhẹ với nhiều đốm nâu phân bố không đều.",
            "Các đốm bệnh màu nâu bắt đầu lan rộng trên phiến lá."
        ],
        "weather": ["rainy", "humid", "cloudy"]
    },
    "LeafBlast": {
        "vietnamese_label": "Đạo ôn lá",
        "symptoms": [
            "vết cháy hình thoi",
            "đầu lá khô",
            "vùng cháy lan rộng",
            "lá chuyển vàng",
            "cháy lá từng mảng"
        ],
        "templates": [
            "Lá lúa xuất hiện các vết cháy hình thoi màu nâu xám.",
            "Triệu chứng bệnh đạo ôn xuất hiện rõ trên phiến lá.",
            "Các đốm cháy màu xám xuất hiện trên lá lúa.",
            "Lá bị bệnh blast với nhiều vết tổn thương kéo dài.",
            "Vết bệnh hình thoi xuất hiện dọc theo phiến lá.",
            "Các vùng cháy màu nâu xám lan rộng trên lá.",
            "Lá có dấu hiệu khô cháy do bệnh đạo ôn."
        ],
        "weather": ["humid", "rainy", "cloudy"]
    },
    "Hispa": {
        "vietnamese_label": "Sâu Hispa",
        "symptoms": [
            "lá bị cào xước",
            "vết trắng trên lá",
            "bề mặt lá hư hại",
            "lá mất màu xanh",
            "sâu ăn biểu bì lá"
        ],
        "templates": [
            "Lá lúa xuất hiện nhiều vệt trắng do sâu ăn phá.",
            "Phiến lá bị sâu Hispa gây hại tạo thành các sọc trắng.",
            "Bề mặt lá xuất hiện các đường trắng kéo dài.",
            "Lá bị côn trùng ăn làm mất lớp biểu bì.",
            "Xuất hiện nhiều vùng bạc màu trên phiến lá.",
            "Lá có dấu hiệu bị sâu phá hoại nghiêm trọng."
        ],
        "weather": ["sunny", "dry", "hot"]
    },
    "Healthy": {
        "vietnamese_label": "Khỏe mạnh",
        "symptoms": [
            "lá xanh khỏe",
            "không có vết bệnh",
            "phiến lá nguyên vẹn",
            "màu lá đồng đều",
            "lá phát triển tốt"
        ],
        "templates": [
            "Lá lúa phát triển khỏe mạnh với màu xanh đều.",
            "Phiến lá xanh đều và không có tổn thương rõ rệt.",
            "Không phát hiện dấu hiệu bệnh trên lá lúa.",
            "Lá có màu xanh tự nhiên và bề mặt sạch.",
            "Lá lúa sinh trưởng tốt và không có tổn thương bệnh lý.",
            "Bề mặt lá nguyên vẹn, không xuất hiện đốm bệnh."
        ],
        "weather": ["sunny", "cloudy", "normal"]
    }
}

LOCATIONS = [
    "Thai Binh",
    "Nam Dinh",
    "Hai Duong",
    "Long An",
    "Soc Trang",
    "Can Tho",
    "An Giang",
    "Dong Thap"
]

SEVERITY_LEVELS = ["mild", "moderate", "severe"]
GROWTH_STAGES = ["seedling", "tillering", "booting", "flowering", "ripening"]
FARMER_NOTES = [
    "Bệnh xuất hiện sau nhiều ngày ẩm ướt.",
    "Triệu chứng lan nhanh trên ruộng.",
    "Độ ẩm cao làm bệnh phát triển mạnh.",
    "Ruộng có dấu hiệu lây lan.",
    "Cây sinh trưởng chậm hơn bình thường.",
    "Nhiều lá xuất hiện triệu chứng tương tự.",
    "Bệnh xuất hiện ở nhiều vị trí trên ruộng.",
    "Tình trạng bệnh tăng nhanh trong tuần qua."
]


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    saturation = np.zeros_like(cmax)
    mask = cmax > 1e-6
    saturation[mask] = delta[mask] / cmax[mask]
    value = cmax
    hue = np.zeros_like(cmax)
    mask_r = (cmax == r) & mask
    mask_g = (cmax == g) & mask
    mask_b = (cmax == b) & mask
    delta_safe = delta + 1e-6
    hue[mask_r] = (60 * ((g - b)[mask_r] / delta_safe[mask_r]) + 360) % 360
    hue[mask_g] = (60 * ((b - r)[mask_g] / delta_safe[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r - g)[mask_b] / delta_safe[mask_b]) + 240) % 360
    return np.stack([hue, saturation, value], axis=-1)


def compute_image_metrics(img_path: Path) -> dict:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    height, width = arr.shape[:2]
    hsv = rgb_to_hsv(arr)
    brightness = hsv[..., 2]
    saturation = hsv[..., 1]

    white_background = (brightness > 0.88) & (saturation < 0.12)
    leaf_mask = ~white_background
    leaf_ratio = float(leaf_mask.mean())

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green_mask = (g > r + 10) & (g > b + 10) & (g > 90)
    brown_mask = (r > 120) & (g > 60) & (b < 110) & (r > g)
    yellow_mask = (r > 150) & (g > 130) & (b < 110) & (r > b)
    white_lesion_mask = (brightness > 0.92) & (saturation < 0.2) & leaf_mask

    green_ratio = float((green_mask & leaf_mask).mean())
    brown_ratio = float((brown_mask & leaf_mask).mean())
    yellow_ratio = float((yellow_mask & leaf_mask).mean())
    white_lesion_ratio = float(white_lesion_mask.mean())

    background_ratio = float(white_background.mean())
    lesion_ratio = brown_ratio + yellow_ratio + white_lesion_ratio
    has_clear_leaf = leaf_ratio > 0.2
    soft_green = float(green_mask.mean())

    return {
        "height": int(height),
        "width": int(width),
        "leaf_area_ratio": round(leaf_ratio, 4),
        "background_ratio": round(background_ratio, 4),
        "green_ratio": round(green_ratio, 4),
        "brown_ratio": round(brown_ratio, 4),
        "yellow_ratio": round(yellow_ratio, 4),
        "white_lesion_ratio": round(white_lesion_ratio, 4),
        "lesion_ratio": round(lesion_ratio, 4),
        "has_clear_leaf": has_clear_leaf,
        "white_background_ratio": float(((r > 220) & (g > 220) & (b > 220)).mean()),
        "average_brightness": float(brightness.mean())
    }


def choose_severity(label: str, lesion_ratio: float) -> str:
    if label == "Healthy":
        return "mild"
    if lesion_ratio > 0.05:
        return "severe"
    if lesion_ratio > 0.02:
        return "moderate"
    return "mild"


def choose_confidence(label: str, metrics: dict) -> float:
    lesion_ratio = metrics["lesion_ratio"]
    if label == "Healthy":
        if lesion_ratio < 0.015:
            return 0.95
        if lesion_ratio < 0.03:
            return 0.82
        return 0.72
    if label == "BrownSpot":
        if lesion_ratio > 0.018:
            return 0.93
        if lesion_ratio > 0.008:
            return 0.86
        return 0.74
    if label == "LeafBlast":
        if lesion_ratio > 0.025:
            return 0.94
        if lesion_ratio > 0.01:
            return 0.85
        return 0.70
    if label == "Hispa":
        if metrics["white_lesion_ratio"] > 0.015:
            return 0.92
        if metrics["white_lesion_ratio"] > 0.008:
            return 0.84
        return 0.72
    return 0.78


def choose_quality(label: str, metrics: dict) -> str:
    confidence = choose_confidence(label, metrics)
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.8:
        return "medium"
    return "low"


def get_visual_observation(label: str, metrics: dict) -> list:
    observations = []
    if metrics["white_background_ratio"] > 0.4:
        observations.append("Phiến lá được chụp rõ ràng trên nền sáng trắng.")
    if metrics["leaf_area_ratio"] > 0.5:
        observations.append("Ảnh chứa phần lớn phiến lá và cho thấy cấu trúc lá rõ ràng.")
    if metrics["leaf_area_ratio"] <= 0.5:
        observations.append("Ảnh chụp cận cảnh một phần phiến lá.")

    if label == "Healthy":
        observations.append("Lá xanh đều, không phát hiện tổn thương bệnh lý rõ ràng.")
    if label == "BrownSpot":
        if metrics["brown_ratio"] >= 0.01:
            observations.append("Trên phiến lá xuất hiện nhiều đốm nâu nhỏ hoặc các vết bệnh rải rác.")
        else:
            observations.append("Lá có dấu hiệu bệnh đốm nâu ở mức độ nhẹ với vết nâu rải rác.")
    if label == "LeafBlast":
        if metrics["brown_ratio"] >= 0.015 or metrics["yellow_ratio"] > 0.005:
            observations.append("Lá có các mảng cháy nâu xám kéo dài điển hình của bệnh đạo ôn.")
        else:
            observations.append("Lá cho thấy triệu chứng vị trí bệnh blast với vết cháy nhẹ.")
    if label == "Hispa":
        if metrics["white_lesion_ratio"] >= 0.01:
            observations.append("Lá xuất hiện nhiều vệt trắng dọc theo chiều dài phiến lá do sâu Hispa.")
        else:
            observations.append("Lá có dấu hiệu tổn thương do sâu làm mất màu xanh và xuất hiện các vệt trắng.")

    if metrics["brown_ratio"] > 0.03:
        observations.append("Các vết nâu/tổn thương chiếm tỷ lệ lớn trên phần lá được quan sát.")
    if metrics["white_lesion_ratio"] > 0.02:
        observations.append("Biểu hiện vết trắng rõ rệt trên bề mặt lá.")
    if not observations:
        observations.append("Hình ảnh cung cấp bằng chứng trực quan cho nhãn hiện tại.")
    return observations


def build_texts(label: str, metrics: dict, num_texts: int = 4) -> list:
    templates = CLASS_INFO[label]["templates"]
    selected = random.sample(templates, k=min(num_texts, len(templates)))
    extra = []
    if metrics["white_background_ratio"] > 0.4:
        extra.append("Lá được chụp trên nền trắng giúp quan sát tổn thương rõ hơn.")
    if label == "BrownSpot" and metrics["brown_ratio"] > 0.015:
        extra.append("Các đốm nâu nhỏ phân bố trên phiến lá và liên kết thành vùng bệnh.")
    if label == "LeafBlast" and metrics["brown_ratio"] > 0.02:
        extra.append("Nhiều mảng cháy màu nâu xám xuất hiện trên phiến lá.")
    if label == "Hispa" and metrics["white_lesion_ratio"] > 0.01:
        extra.append("Các sọc trắng dài do sâu ăn biểu hiện rõ trên lá.")
    if label == "Healthy" and metrics["green_ratio"] > 0.4:
        extra.append("Lá vẫn giữ màu xanh tươi và không có đốm bệnh đáng kể.")
    if metrics["leaf_area_ratio"] <= 0.4:
        extra.append("Ảnh cho thấy cận cảnh phần lá, nhấn mạnh tổn thương và cấu trúc mô lá.")
    texts = selected[:3] + extra
    return texts[:max(3, min(len(texts), 5))]


def generate_context(label: str) -> dict:
    return {
        "weather": random.choice(CLASS_INFO[label]["weather"]),
        "humidity": random.randint(60, 95),
        "temperature": random.randint(22, 36),
        "severity": None,
        "growth_stage": random.choice(GROWTH_STAGES),
        "location": random.choice(LOCATIONS),
        "farmer_note": random.choice(FARMER_NOTES)
    }


def main():
    samples = []
    for label in sorted([p.name for p in DATASET_ROOT.iterdir() if p.is_dir()]):
        class_dir = DATASET_ROOT / label
        image_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        for img_path in image_files:
            metrics = compute_image_metrics(img_path)
            context = generate_context(label)
            context["severity"] = choose_severity(label, metrics["lesion_ratio"])
            confidence = choose_confidence(label, metrics)
            quality = choose_quality(label, metrics)
            visual_obs = get_visual_observation(label, metrics)
            texts = build_texts(label, metrics, num_texts=4)
            sample = {
                "image": str(Path("dataset/raw") / label / img_path.name).replace("\\", "/"),
                "label": label,
                "vietnamese_label": CLASS_INFO[label]["vietnamese_label"],
                "texts": texts,
                "symptoms": random.sample(CLASS_INFO[label]["symptoms"], k=min(4, len(CLASS_INFO[label]["symptoms"]))),
                "visual_analysis": visual_obs,
                "leaf_area_ratio": metrics["leaf_area_ratio"],
                "background_ratio": metrics["background_ratio"],
                "lesion_area_ratio": metrics["lesion_ratio"],
                "annotation_confidence": confidence,
                "metadata_quality": quality,
                "weather": context["weather"],
                "humidity": context["humidity"],
                "temperature": context["temperature"],
                "severity": context["severity"],
                "growth_stage": context["growth_stage"],
                "location": context["location"],
                "farmer_note": context["farmer_note"]
            }
            samples.append(sample)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    rows = []
    for sample in samples:
        rows.append({
            "image": sample["image"],
            "label": sample["label"],
            "vietnamese_label": sample["vietnamese_label"],
            "texts": " | ".join(sample["texts"]),
            "symptoms": " | ".join(sample["symptoms"]),
            "visual_analysis": " | ".join(sample["visual_analysis"]),
            "leaf_area_ratio": sample["leaf_area_ratio"],
            "background_ratio": sample["background_ratio"],
            "lesion_area_ratio": sample["lesion_area_ratio"],
            "annotation_confidence": sample["annotation_confidence"],
            "metadata_quality": sample["metadata_quality"],
            "weather": sample["weather"],
            "humidity": sample["humidity"],
            "temperature": sample["temperature"],
            "severity": sample["severity"],
            "growth_stage": sample["growth_stage"],
            "location": sample["location"],
            "farmer_note": sample["farmer_note"]
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "total_samples": len(samples),
        "classes": dict(Counter([sample["label"] for sample in samples])),
        "metadata_fields": [
            "image",
            "label",
            "vietnamese_label",
            "texts",
            "symptoms",
            "visual_analysis",
            "leaf_area_ratio",
            "background_ratio",
            "lesion_area_ratio",
            "annotation_confidence",
            "metadata_quality",
            "weather",
            "humidity",
            "temperature",
            "severity",
            "growth_stage",
            "location",
            "farmer_note"
        ]
    }
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
