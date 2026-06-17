import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from datetime import datetime
import random


LABEL2ID = {
    "BrownSpot": 0,
    "Healthy": 1,
    "Hispa": 2,
    "LeafBlast": 3,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Mapping cho categorical features
WEATHER_MAP = {"sunny": 0, "cloudy": 1, "rainy": 2}
SEVERITY_MAP = {"low": 0, "moderate": 1, "high": 2}
GROWTH_STAGE_MAP = {"seedling": 0, "tillering": 1, "booting": 2, "flowering": 3, "ripening": 4}
LOCATION_MAP = {"Dong Thap": 0, "An Giang": 1, "Can Tho": 2, "Long An": 3, "other": 4}

METADATA_FIELDS = [
    "weather", "humidity", "temperature", "severity", 
    "growth_stage", "location", "farmer_note",
    "leaf_area_ratio", "background_ratio", "lesion_area_ratio", "annotation_confidence"
]

# Cấu hình crop động
MIN_VALID_COVERAGE = 0.05
MAX_CROP_RATIO = 0.85
FALLBACK_PADDING = 0.1


def extract_leaf_metrics_from_image(image_path: Path, target_size: int = 512):
    """
    Trích xuất hull, mask, coverage từ ảnh gốc
    Trả về: (coverage, aspect_ratio, has_valid_leaf, hull_points, hull_mask)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.5, 1.0, False, None, None
        
        h, w = img.shape[:2]
        
        # Resize để xử lý nhanh hơn
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            h, w = img.shape[:2]
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mask_green = cv2.inRange(hsv, np.array([25, 30, 20]), np.array([95, 255, 255]))
        mask_brown = cv2.inRange(hsv, np.array([5, 20, 20]), np.array([25, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([15, 20, 40]), np.array([40, 255, 255]))
        
        mask = cv2.bitwise_or(mask_green, mask_brown)
        mask = cv2.bitwise_or(mask, mask_yellow)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return 0.5, 1.0, False, None, None
        
        min_area = max(200, int(0.001 * w * h))
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if len(valid_contours) == 0:
            return 0.5, 1.0, False, None, None
        
        all_points = np.vstack(valid_contours)
        hull = cv2.convexHull(all_points)
        hull_area = cv2.contourArea(hull)
        
        coverage = hull_area / (w * h)
        
        if coverage > 0.50:
            largest = max(valid_contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            coverage = hull_area / (w * h)
        
        rect = cv2.minAreaRect(hull)
        rw, rh = rect[1]
        if min(rw, rh) > 0:
            aspect_ratio = max(rw, rh) / min(rw, rh)
            aspect_ratio = min(aspect_ratio, 10.0)
        else:
            aspect_ratio = 1.0
        
        # Tạo mask từ hull
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        
        hull_points = hull.reshape(-1, 2)
        
        return coverage, aspect_ratio, True, hull_points.tolist(), hull_mask
        
    except Exception as e:
        print(f"Warning: Error extracting leaf metrics from {image_path}: {e}")
        return 0.5, 1.0, False, None, None


class RiceDiseaseDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        tokenizer_name="vinai/phobert-base",
        max_length=128,
        transform=None,
        use_symptoms=True,
        use_visual_analysis=True,
        metadata_fields=None,
        deterministic_text=True,
        use_metadata=True,
        extract_leaf_metrics=False,
        image_size_for_extraction=512,
        use_leaf_mask=True,  # Bật/tắt mask
    ):
        self.metadata_path = Path(metadata_path)
        self.transform = transform
        self.max_length = max_length
        self.use_symptoms = use_symptoms
        self.use_visual_analysis = use_visual_analysis
        self.metadata_fields = metadata_fields or METADATA_FIELDS
        self.deterministic_text = deterministic_text
        self.use_metadata = use_metadata
        self.extract_leaf_metrics = extract_leaf_metrics
        self.image_size_for_extraction = image_size_for_extraction
        self.use_leaf_mask = use_leaf_mask

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.deterministic_text:
            self.processed_texts = [self._build_text(sample) for sample in self.samples]
        
        self.leaf_metrics_cache = {}
        
        if self.extract_leaf_metrics:
            self._precompute_leaf_metrics()
    
    def _precompute_leaf_metrics(self):
        """Pre-compute leaf metrics cho tất cả samples"""
        print(f"Pre-computing leaf metrics for {len(self.samples)} samples...")
        from tqdm import tqdm
        
        success_count = 0
        for idx, sample in enumerate(tqdm(self.samples, desc="Extracting leaf metrics")):
            image_path = Path(sample["image"])
            if image_path.exists():
                coverage, aspect_ratio, has_valid, hull_points, hull_mask = extract_leaf_metrics_from_image(
                    image_path, self.image_size_for_extraction
                )
                self.leaf_metrics_cache[idx] = {
                    'coverage': coverage,
                    'aspect_ratio': aspect_ratio,
                    'has_valid_leaf': has_valid,
                    'hull_points': hull_points,
                    'hull_mask': hull_mask,
                }
                if has_valid:
                    success_count += 1
            else:
                self.leaf_metrics_cache[idx] = {
                    'coverage': 0.5,
                    'aspect_ratio': 1.0,
                    'has_valid_leaf': False,
                    'hull_points': None,
                    'hull_mask': None,
                }
        
        total = len(self.samples)
        print(f"\n📊 Leaf Metrics Statistics:")
        print(f"  ✅ Valid masks: {success_count}/{total} ({success_count/total*100:.1f}%)")
        print("Done pre-computing leaf metrics!")
    
    def __len__(self):
        return len(self.samples)

    def _build_text(self, sample):
        texts = sample.get("texts", [])
        symptoms = sample.get("symptoms", [])
        visual_analysis = sample.get("visual_analysis", [])

        text_parts = []
        
        if texts:
            text_parts.extend(texts)
            
        if self.use_symptoms and symptoms:
            text_parts.extend(symptoms)
            
        if self.use_visual_analysis and visual_analysis:
            text_parts.extend(visual_analysis)
        
        separator = "\n[SEP]\n"
        return separator.join(text_parts).strip()

    def _encode_metadata(self, sample, leaf_metrics=None):
        """Encode metadata + leaf metrics thành tensor features (26 dimensions)"""
        features = []
        
        weather = sample.get("weather", "sunny")
        weather_onehot = torch.zeros(3)
        weather_onehot[WEATHER_MAP.get(weather, 0)] = 1
        features.append(weather_onehot)
        
        humidity = sample.get("humidity", 70)
        features.append(torch.tensor([humidity / 100.0]))
        
        temp = sample.get("temperature", 25)
        features.append(torch.tensor([(temp - 15) / 20.0]))
        
        severity = sample.get("severity", "moderate")
        severity_onehot = torch.zeros(3)
        severity_onehot[SEVERITY_MAP.get(severity, 1)] = 1
        features.append(severity_onehot)
        
        growth = sample.get("growth_stage", "tillering")
        growth_onehot = torch.zeros(5)
        growth_onehot[GROWTH_STAGE_MAP.get(growth, 1)] = 1
        features.append(growth_onehot)
        
        location = sample.get("location", "Dong Thap")
        location_onehot = torch.zeros(5)
        location_onehot[LOCATION_MAP.get(location, 4)] = 1
        features.append(location_onehot)
        
        farmer_note = sample.get("farmer_note", "")
        features.append(torch.tensor([min(len(farmer_note) / 200.0, 1.0)]))
        
        if leaf_metrics:
            leaf_coverage = leaf_metrics.get('coverage', 0.5)
            aspect_ratio = leaf_metrics.get('aspect_ratio', 1.0)
            has_valid_leaf = 1.0 if leaf_metrics.get('has_valid_leaf', False) else 0.0
        else:
            leaf_coverage = sample.get("leaf_area_ratio", 0.1)
            aspect_ratio = 1.0
            has_valid_leaf = 1.0
        
        features.append(torch.tensor([min(leaf_coverage, 1.0)]))
        features.append(torch.tensor([min(aspect_ratio, 5.0) / 5.0]))
        features.append(torch.tensor([has_valid_leaf]))
        
        leaf_area = sample.get("leaf_area_ratio", 0.1)
        features.append(torch.tensor([min(leaf_area, 1.0)]))
        background_area = sample.get("background_ratio", 0.9)
        features.append(torch.tensor([min(background_area, 1.0)]))
        lesion_area = sample.get("lesion_area_ratio", 0.05)
        features.append(torch.tensor([min(lesion_area, 1.0)]))
        
        confidence = sample.get("annotation_confidence", 0.8)
        features.append(torch.tensor([confidence]))
        
        metadata_tensor = torch.cat(features)
        return metadata_tensor

    def _resize_mask(self, mask, target_size):
        """Resize mask về target_size, giữ nguyên giá trị nhị phân"""
        if mask is None:
            return None
        return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image_path = Path(sample["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        
        # Lấy leaf metrics từ cache
        leaf_metrics = self.leaf_metrics_cache.get(idx) if self.extract_leaf_metrics else None
        
        # Lấy target size từ transform
        if self.transform:
            # Lưu target size trước khi transform
            target_size = (224, 224)  # Mặc định
            image = self.transform(image)
        else:
            target_size = image.size
        
        # Resize mask về cùng kích thước với ảnh đã transform
        leaf_mask = None
        if self.use_leaf_mask and leaf_metrics and leaf_metrics.get('hull_mask') is not None:
            hull_mask = leaf_metrics['hull_mask']
            if hull_mask is not None:
                leaf_mask = self._resize_mask(hull_mask, target_size)
                # Chuyển thành tensor float [0,1]
                leaf_mask = torch.from_numpy(leaf_mask).float() / 255.0
        
        # Process text
        if self.deterministic_text:
            text = self.processed_texts[idx]
        else:
            text = self._build_text(sample)
            
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Process metadata
        metadata_tensor = None
        if self.use_metadata:
            metadata_tensor = self._encode_metadata(sample, leaf_metrics)

        label = LABEL2ID[sample["label"]]

        result = {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask_text": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": text,
            "image_path": str(sample["image"]),
            "metadata": metadata_tensor,
        }
        
        if leaf_metrics:
            result["leaf_coverage"] = torch.tensor(leaf_metrics['coverage'], dtype=torch.float)
            result["leaf_aspect_ratio"] = torch.tensor(leaf_metrics['aspect_ratio'], dtype=torch.float)
            result["has_valid_leaf"] = torch.tensor(leaf_metrics['has_valid_leaf'], dtype=torch.float)
        
        # Thêm leaf mask vào output - QUAN TRỌNG
        if leaf_mask is not None:
            result["leaf_mask"] = leaf_mask  # Mask riêng, không nhân với ảnh
        
        return result


def main_visualize_samples():
    """
    Hàm main để test và xuất ảnh hull & mask cho mỗi nhãn 3 mẫu
    """
    print("=" * 80)
    print("🌾 VISUALIZE HULL & MASK AFTER RESIZE")
    print("=" * 80)
    print("\nMục đích: Kiểm tra hull và mask sau khi resize lên 224x224")
    print("  - Ảnh gốc → Resize về 224x224")
    print("  - Hull mask → Resize về 224x224")
    print("  - Vẽ hull (màu đỏ) lên ảnh đã resize")
    print("=" * 80)
    
    dataset_root = Path("dataset/raw")
    labels = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]
    samples_per_label = 3
    
    output_dir = Path("hull_mask_visualization")
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for label in labels:
        label_path = dataset_root / label
        if not label_path.exists():
            print(f"⚠️ Path not found: {label_path}")
            continue
        
        images = list(label_path.glob("*.jpg")) + list(label_path.glob("*.png"))
        print(f"\n📁 {label}: found {len(images)} images")
        
        random.seed(42)
        selected = random.sample(images, min(samples_per_label, len(images)))
        
        for idx, img_path in enumerate(selected):
            print(f"  Processing sample {idx+1}/{len(selected)}: {img_path.name}")
            
            result = extract_leaf_metrics_from_image(img_path)
            coverage, aspect_ratio, has_valid, hull_points, hull_mask = result
            
            if hull_mask is None:
                print(f"    ⚠️ Cannot process")
                continue
            
            original = cv2.imread(str(img_path))
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            h, w = original.shape[:2]
            
            target_size = 224
            resized_img = cv2.resize(original, (target_size, target_size))
            resized_mask = cv2.resize(hull_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            contours_resized, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{label} - {img_path.name}", fontsize=14, fontweight='bold')
            
            axes[0, 0].imshow(original_rgb)
            axes[0, 0].set_title(f"Original Image\nSize: {w}x{h}\nCoverage: {coverage:.2%}")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(hull_mask, cmap='gray')
            axes[0, 1].set_title(f"Hull Mask (Original)\nAspect Ratio: {aspect_ratio:.1f}")
            axes[0, 1].axis('off')
            
            img_with_hull = resized_img.copy()
            cv2.drawContours(img_with_hull, contours_resized, -1, (0, 0, 255), 2)
            axes[1, 0].imshow(cv2.cvtColor(img_with_hull, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title(f"Resized Image (224x224) with Hull Overlay")
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(resized_mask, cmap='gray')
            axes[1, 1].set_title(f"Resized Mask (224x224)")
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            save_path = output_dir / f"{idx+1:02d}_{label}_{img_path.stem}_hull_mask.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            all_results.append({
                'label': label,
                'image': img_path.name,
                'coverage': coverage,
                'aspect_ratio': aspect_ratio,
            })
            print(f"    ✓ Coverage: {coverage:.2%}, Aspect Ratio: {aspect_ratio:.1f}")
    
    report_path = output_dir / "visualization_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(all_results),
            'samples': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETED!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main_visualize_samples()