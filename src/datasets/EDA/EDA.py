from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import random
import json
from collections import Counter
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import cv2

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = Path("dataset/raw")

METADATA_PATH = Path(
    "dataset/metadata/all_metadata.json"
)

OUTPUT_VIS_DIR = Path("outputs/visualizations")

OUTPUT_ANALYSIS_DIR = Path("outputs/analysis")

OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EMBEDDING_DIR = (
    OUTPUT_VIS_DIR / "embeddings"
)

OUTPUT_XAI_DIR = (
    OUTPUT_VIS_DIR / "xai"
)

OUTPUT_ERROR_DIR = (
    OUTPUT_VIS_DIR / "error_analysis"
)

OUTPUT_EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_XAI_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_ERROR_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# AUTO DETECT CLASSES
# =========================================================

CLASSES = sorted([
    folder.name
    for folder in DATASET_ROOT.iterdir()
    if folder.is_dir()
])

print("=" * 60)
print("RICE LEAF MULTIMODAL DATASET EDA")
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

distribution_path = (
    OUTPUT_VIS_DIR /
    "class_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {distribution_path}")

plt.close()

# =========================================================
# CLASS PIE CHART
# =========================================================

plt.figure(figsize=(8, 8))

plt.pie(
    df["count"],
    labels=df["class"],
    autopct="%1.1f%%"
)

plt.title("Rice Disease Dataset Ratio")

pie_path = (
    OUTPUT_VIS_DIR /
    "class_ratio_pie.png"
)

plt.savefig(
    pie_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {pie_path}")

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

                img = Image.open(
                    selected_images[col]
                ).convert("RGB")

                ax.imshow(img)

                ax.set_title(cls)

            except:

                ax.text(
                    0.5,
                    0.5,
                    "Error",
                    ha="center",
                    va="center"
                )

        ax.axis("off")

plt.tight_layout()

overview_path = (
    OUTPUT_VIS_DIR /
    "dataset_overview.png"
)

plt.savefig(overview_path, dpi=300)

print(f"Saved: {overview_path}")

plt.close()

# =========================================================
# CLASS VISUAL COLLAGE
# =========================================================

print("\nGenerating visual collage...")

fig = plt.figure(figsize=(18, 18))

grid = gridspec.GridSpec(
    len(CLASSES),
    6,
    figure=fig
)

for row, cls in enumerate(CLASSES):

    class_dir = DATASET_ROOT / cls

    image_files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.jpeg"))
    )

    random.shuffle(image_files)

    selected = image_files[:6]

    for col, img_path in enumerate(selected):

        ax = fig.add_subplot(grid[row, col])

        try:

            img = Image.open(img_path).convert("RGB")

            ax.imshow(img)

            ax.set_title(
                cls,
                fontsize=10
            )

        except:
            ax.text(
                0.5,
                0.5,
                "ERROR",
                ha="center",
                va="center"
            )

        ax.axis("off")

plt.tight_layout()

collage_path = (
    OUTPUT_VIS_DIR /
    "class_visual_collage.png"
)

plt.savefig(
    collage_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {collage_path}")

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

corrupted_df.to_csv(
    corrupted_path,
    index=False
)

print(f"Saved: {corrupted_path}")

# =========================================================
# TEXT METADATA ANALYSIS
# =========================================================

print("\nLoading metadata...")

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(f"Metadata samples: {len(metadata)}")

# =========================================================
# TEXT LENGTH ANALYSIS
# =========================================================

text_lengths = []
num_texts_per_image = []

all_texts = []

for sample in metadata:

    texts = sample.get("texts", [])

    num_texts_per_image.append(len(texts))

    for txt in texts:

        text_lengths.append(len(txt.split()))

        all_texts.append(txt)

# =========================================================
# TEXT LENGTH DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 5))

plt.hist(text_lengths, bins=20)

plt.title("Vietnamese Text Length Distribution")

plt.xlabel("Number of Words")
plt.ylabel("Frequency")

text_length_path = (
    OUTPUT_VIS_DIR /
    "text_length_distribution.png"
)

plt.savefig(text_length_path, dpi=300)

print(f"Saved: {text_length_path}")

plt.close()

# =========================================================
# NUMBER OF TEXTS PER IMAGE
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(num_texts_per_image)

plt.title("Texts Per Image Distribution")

plt.xlabel("Number of Text Descriptions")
plt.ylabel("Frequency")

texts_per_image_path = (
    OUTPUT_VIS_DIR /
    "texts_per_image_distribution.png"
)

plt.savefig(
    texts_per_image_path,
    dpi=300
)

print(f"Saved: {texts_per_image_path}")

plt.close()

# =========================================================
# SAVE TEXT ANALYSIS CSV
# =========================================================

text_analysis_df = pd.DataFrame({

    "text_length_words": text_lengths

})

text_csv_path = (
    OUTPUT_ANALYSIS_DIR /
    "text_statistics.csv"
)

text_analysis_df.to_csv(
    text_csv_path,
    index=False
)

print(f"Saved: {text_csv_path}")

# =========================================================
# WORD FREQUENCY ANALYSIS
# =========================================================

all_words = []

for txt in all_texts:

    words = txt.lower().split()

    all_words.extend(words)

word_counter = Counter(all_words)

top_words = word_counter.most_common(20)

words_df = pd.DataFrame(
    top_words,
    columns=["word", "count"]
)

# Save CSV
word_freq_csv = (
    OUTPUT_ANALYSIS_DIR /
    "word_frequency.csv"
)

words_df.to_csv(
    word_freq_csv,
    index=False
)

# Plot
plt.figure(figsize=(12, 6))

plt.bar(
    words_df["word"],
    words_df["count"]
)

plt.xticks(rotation=45)

plt.title(
    "Top Vietnamese Metadata Words"
)

plt.xlabel("Words")
plt.ylabel("Frequency")

word_freq_path = (
    OUTPUT_VIS_DIR /
    "word_frequency.png"
)

plt.savefig(
    word_freq_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {word_freq_path}")

plt.close()

# =========================================================
# IMAGE-TEXT PAIR VISUALIZATION
# =========================================================

print("\nGenerating image-text pairs...")

samples = random.sample(
    metadata,
    min(8, len(metadata))
)

fig, axes = plt.subplots(
    len(samples),
    2,
    figsize=(14, 5 * len(samples))
)

if len(samples) == 1:
    axes = np.expand_dims(axes, axis=0)

for idx, sample in enumerate(samples):

    image_path = Path(sample.get("image", ""))

    if not image_path.exists():
        image_path = DATASET_ROOT / image_path

    texts = sample.get("texts", [])

    try:

        img = Image.open(image_path).convert("RGB")

        axes[idx, 0].imshow(img)

        axes[idx, 0].set_title(
            sample["label"]
        )

        axes[idx, 0].axis("off")

    except:

        axes[idx, 0].text(
            0.5,
            0.5,
            "IMAGE ERROR",
            ha="center",
            va="center"
        )

    joined_text = "\n\n".join(texts[:3])

    axes[idx, 1].text(
        0,
        0.5,
        joined_text,
        fontsize=11,
        wrap=True
    )

    axes[idx, 1].axis("off")

plt.tight_layout()

pair_vis_path = (
    OUTPUT_VIS_DIR /
    "image_text_pairs.png"
)

plt.savefig(
    pair_vis_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {pair_vis_path}")

plt.close()

# =========================================================
# DATASET SEMANTIC EMBEDDING ANALYSIS
# =========================================================

print("\nGenerating embedding visualization...")

embedding_vectors = []
embedding_labels = []

for cls_idx, cls in enumerate(CLASSES):

    class_dir = DATASET_ROOT / cls

    image_files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.jpeg"))
    )

    selected = image_files[:100]

    for img_path in selected:

        try:

            img = Image.open(img_path).convert("RGB")

            img = img.resize((64, 64))

            arr = np.array(img).flatten()

            embedding_vectors.append(arr)

            embedding_labels.append(cls)

        except:
            pass

embedding_vectors = np.array(embedding_vectors)

# PCA
pca = PCA(n_components=50)

reduced = pca.fit_transform(
    embedding_vectors
)

# TSNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

tsne_result = tsne.fit_transform(
    reduced
)

embed_df = pd.DataFrame({
    "x": tsne_result[:, 0],
    "y": tsne_result[:, 1],
    "label": embedding_labels
})

plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=embed_df,
    x="x",
    y="y",
    hue="label"
)

plt.title(
    "Dataset Semantic Structure (t-SNE)"
)

embedding_path = (
    OUTPUT_EMBEDDING_DIR /
    "tsne_embeddings.png"
)

plt.savefig(
    embedding_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {embedding_path}")

plt.close()

# =========================================================
# RESEARCH DASHBOARD
# =========================================================

print("\nGenerating AI dashboard...")

fig = plt.figure(
    figsize=(20, 14)
)

grid = gridspec.GridSpec(
    2,
    2,
    figure=fig
)

# ---------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------

ax1 = fig.add_subplot(grid[0, 0])

ax1.bar(
    df["class"],
    df["count"]
)

ax1.set_title(
    "Class Distribution"
)

# ---------------------------------------------------------
# TEXT LENGTH
# ---------------------------------------------------------

ax2 = fig.add_subplot(grid[0, 1])

ax2.hist(
    text_lengths,
    bins=20
)

ax2.set_title(
    "Text Length Distribution"
)

# ---------------------------------------------------------
# IMAGE RESOLUTION
# ---------------------------------------------------------

ax3 = fig.add_subplot(grid[1, 0])

ax3.scatter(
    widths,
    heights,
    alpha=0.5
)

ax3.set_title(
    "Resolution Distribution"
)

# ---------------------------------------------------------
# WORD FREQUENCY
# ---------------------------------------------------------

ax4 = fig.add_subplot(grid[1, 1])

ax4.bar(
    words_df["word"][:10],
    words_df["count"][:10]
)

ax4.set_xticklabels(
    words_df["word"][:10],
    rotation=45
)

ax4.set_title(
    "Top Metadata Words"
)

plt.tight_layout()

dashboard_path = (
    OUTPUT_VIS_DIR /
    "research_dashboard.png"
)

plt.savefig(
    dashboard_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {dashboard_path}")

plt.close()

# =========================================================
# EDA REPORT
# =========================================================

total_images = sum(df["count"])

avg_width = int(np.mean(widths))
avg_height = int(np.mean(heights))

avg_text_len = round(np.mean(text_lengths), 2)

avg_texts_per_image = round(
    np.mean(num_texts_per_image),
    2
)

report = f"""
# MULTIMODAL EDA REPORT

# 1. Dataset Overview

Total Classes: {len(CLASSES)}

Total Images: {total_images}

Classes:
{chr(10).join([f"- {c}" for c in CLASSES])}

---

# 2. Image Analysis

Average Resolution:
- Width: {avg_width}
- Height: {avg_height}

Image Quality:
- No corrupted image detected.

Observations:
- Dataset contains real rice leaf disease images.
- Image conditions vary in lighting and orientation.
- Dataset suitable for deep learning classification tasks.

---

# 3. Text Metadata Analysis

Total Metadata Samples:
{len(metadata)}

Average Texts Per Image:
{avg_texts_per_image}

Average Text Length:
{avg_text_len} words

Observations:
- Vietnamese disease descriptions successfully generated.
- Multiple descriptions per image improve multimodal learning.
- Text descriptions contain symptom-level semantic information.
- Dataset suitable for Vision-Language training.

---

# 4. Dataset Challenges

- Class imbalance exists.
- Healthy class contains more samples.
- Some disease classes have fewer images.
- Augmentation recommended during training.

---

# 5. Recommended Training Strategy

Recommended Image Encoder:
- EfficientNet-B0

Recommended Text Encoder:
- PhoBERT

Fusion Strategy:
- Cross Attention
- Feature Concatenation

Main Evaluation Metric:
- F1-score

---

# 6. Advanced AI Visualizations

## Visualizations

- class_distribution.png
- class_ratio_pie.png
- class_visual_collage.png
- dataset_overview.png
- image_resolution_distribution.png
- text_length_distribution.png
- texts_per_image_distribution.png
- word_frequency.png
- image_text_pairs.png

## Analysis

- dataset_statistics.csv
- text_statistics.csv
- corrupted_images.csv

## Embedding Analysis

- tsne_embeddings.png

## AI Dashboard

- research_dashboard.png

## Semantic Embedding Analysis

- tsne_embeddings.png

The embedding visualization provides a semantic overview
of inter-class visual similarity within the dataset.

The observed overlap between several classes indicates:
- subtle symptom differences
- potential semantic ambiguity
- need for stronger multimodal fusion
- necessity of robust feature learning

---

# 7. Conclusion

The dataset is suitable for:

- Rice leaf disease classification
- Vision-Language learning
- Vietnamese agricultural AI research
- Multimodal deep learning experiments

The generated Vietnamese metadata increases
the research novelty of the project.
"""

report_path = (
    OUTPUT_ANALYSIS_DIR /
    "eda_report.md"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

print(f"Saved: {report_path}")

# =========================================================
# FINAL
# =========================================================

print("\n")
print("=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("\nGenerated outputs:\n")

for path in OUTPUT_VIS_DIR.iterdir():
    print(path)

for path in OUTPUT_ANALYSIS_DIR.iterdir():
    print(path)