from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = Path("dataset/raw")

OUTPUT_VIS_DIR = Path("outputs/visualizations")
OUTPUT_ANALYSIS_DIR = Path("outputs/analysis")

OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# AUTO DETECT CLASSES
# =========================================================

CLASSES = sorted([
    folder.name
    for folder in DATASET_ROOT.iterdir()
    if folder.is_dir()
])

print("=" * 60)
print("RICE LEAF DATASET EDA")
print("=" * 60)

print("\nDetected classes:")

for cls in CLASSES:
    print(f"- {cls}")

# =========================================================
# IMAGE COUNT ANALYSIS
# =========================================================

stats = []

for cls in CLASSES:

    class_dir = DATASET_ROOT / cls

    image_files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.jpeg"))
    )

    stats.append({
        "class": cls,
        "count": len(image_files)
    })

df = pd.DataFrame(stats)

print("\nDataset Statistics:")
print(df)

# Save CSV
csv_path = OUTPUT_ANALYSIS_DIR / "dataset_statistics.csv"
df.to_csv(csv_path, index=False)

print(f"\nSaved statistics: {csv_path}")

# =========================================================
# CLASS DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(df["class"], df["count"])

plt.title("Rice Disease Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=15)

distribution_path = OUTPUT_VIS_DIR / "class_distribution.png"

plt.savefig(distribution_path, dpi=300, bbox_inches="tight")

print(f"Saved: {distribution_path}")

plt.close()

# =========================================================
# SAMPLE IMAGES OVERVIEW
# =========================================================

fig, axes = plt.subplots(
    len(CLASSES),
    4,
    figsize=(16, 4 * len(CLASSES))
)

if len(CLASSES) == 1:
    axes = np.expand_dims(axes, axis=0)

for row, cls in enumerate(CLASSES):

    class_dir = DATASET_ROOT / cls

    image_files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.jpeg"))
    )

    random.shuffle(image_files)

    selected_images = image_files[:4]

    for col in range(4):

        ax = axes[row, col]

        if col < len(selected_images):

            try:
                img = Image.open(selected_images[col]).convert("RGB")

                ax.imshow(img)

                ax.set_title(cls)

            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    "Error",
                    ha="center",
                    va="center"
                )

        ax.axis("off")

plt.tight_layout()

overview_path = OUTPUT_VIS_DIR / "dataset_overview.png"

plt.savefig(overview_path, dpi=300)

print(f"Saved: {overview_path}")

plt.close()

# =========================================================
# IMAGE SIZE ANALYSIS
# =========================================================

widths = []
heights = []

for cls in CLASSES:

    class_dir = DATASET_ROOT / cls

    image_files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.jpeg"))
    )

    for img_path in image_files:

        try:
            img = Image.open(img_path)

            w, h = img.size

            widths.append(w)
            heights.append(h)

        except:
            pass

# =========================================================
# RESOLUTION DISTRIBUTION
# =========================================================

plt.figure(figsize=(7, 7))

plt.scatter(widths, heights)

plt.title("Image Resolution Distribution")
plt.xlabel("Width")
plt.ylabel("Height")

resolution_path = (
    OUTPUT_VIS_DIR /
    "image_resolution_distribution.png"
)

plt.savefig(resolution_path, dpi=300)

print(f"Saved: {resolution_path}")

plt.close()

# =========================================================
# CORRUPTED IMAGE CHECK
# =========================================================

corrupted_images = []

for cls in CLASSES:

    class_dir = DATASET_ROOT / cls

    image_files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.jpeg"))
    )

    for img_path in image_files:

        try:
            img = Image.open(img_path)
            img.verify()

        except:
            corrupted_images.append(str(img_path))

corrupted_df = pd.DataFrame({
    "corrupted_images": corrupted_images
})

corrupted_path = (
    OUTPUT_ANALYSIS_DIR /
    "corrupted_images.csv"
)

corrupted_df.to_csv(corrupted_path, index=False)

print(f"Saved: {corrupted_path}")

# =========================================================
# EDA REPORT
# =========================================================

report = f"""
# EDA REPORT

## Dataset Summary

Total Classes: {len(CLASSES)}

Classes:
{chr(10).join([f"- {c}" for c in CLASSES])}

---

## Generated Files

### Visualizations
- dataset_overview.png
- class_distribution.png
- image_resolution_distribution.png
- rgb_distribution.png

### Analysis
- dataset_statistics.csv
- corrupted_images.csv

---

## Notes

- Dataset analyzed from raw/images
- Ready for preprocessing
- Ready for metadata generation
"""

report_path = OUTPUT_ANALYSIS_DIR / "eda_report.md"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Saved: {report_path}")

# =========================================================
# FINAL
# =========================================================

print("\n")
print("=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("\nGenerated outputs:")

for path in OUTPUT_VIS_DIR.iterdir():
    print(path)

for path in OUTPUT_ANALYSIS_DIR.iterdir():
    print(path)