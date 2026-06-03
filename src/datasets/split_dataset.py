import json
import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_metadata(metadata_path):
    """Load metadata từ JSON hoặc CSV file"""
    metadata_path = Path(metadata_path)
    
    if metadata_path.suffix == '.json':
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif metadata_path.suffix == '.csv':
        df = pd.read_csv(metadata_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {metadata_path.suffix}")


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(data, path):
    """Save list of dicts to CSV"""
    if not data:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding='utf-8')


def split_dataset(metadata_path, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split dataset thành train/val/test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    print(f"Loading metadata from: {metadata_path}")
    metadata = load_metadata(metadata_path)
    print(f"Total samples: {len(metadata)}")
    
    # Extract labels
    labels = [sample["label"] for sample in metadata]
    
    # First split: train vs temp
    train_data, temp_data = train_test_split(
        metadata,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: val vs test
    temp_labels = [sample["label"] for sample in temp_data]
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_state,
        stratify=temp_labels
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    save_json(train_data, output_dir / "train.json")
    save_json(val_data, output_dir / "val.json")
    save_json(test_data, output_dir / "test.json")
    
    # Also save as CSV for easy viewing
    save_csv(train_data, output_dir / "train.csv")
    save_csv(val_data, output_dir / "val.csv")
    save_csv(test_data, output_dir / "test.csv")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train: {len(train_data)} samples ({len(train_data)/len(metadata)*100:.1f}%)")
    print(f"Val:   {len(val_data)} samples ({len(val_data)/len(metadata)*100:.1f}%)")
    print(f"Test:  {len(test_data)} samples ({len(test_data)/len(metadata)*100:.1f}%)")
    print("=" * 60)
    
    # Print class distribution
    print("\nClass distribution:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        print(f"\n{split_name}:")
        label_counts = {}
        for sample in split_data:
            label = sample["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # =========================================================================
    # CẤU HÌNH TRỰC TIẾP TẠI ĐÂY
    # =========================================================================
    METADATA_PATH = "dataset/metadata/all_metadata.json"  # Thay bằng đường dẫn file JSON hoặc CSV của bạn
    OUTPUT_DIR = "dataset/splits"        # Thư mục lưu kết quả sau khi split
    
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    SEED = 42
    # =========================================================================

    # Chạy hàm split dataset với cấu hình trên
    split_dataset(
        metadata_path=METADATA_PATH,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=SEED
    )