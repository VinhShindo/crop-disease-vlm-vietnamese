import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image, ImageStat


def _load_metadata(metadata_path):
    p = Path(metadata_path)
    if p.suffix == '.json':
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif p.suffix == '.csv':
        return pd.read_csv(p)
    else:
        raise ValueError('Unsupported metadata format: ' + str(p))


def plot_class_distribution(df, class_col='label', save_path=None):
    counts = df[class_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class distribution')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_metadata_correlation(df, numeric_cols, save_path=None):
    if not numeric_cols:
        return
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(numeric_cols, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Metadata correlation')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_missing_matrix(df, save_path=None):
    missing = df.isnull().astype(int)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(missing.T, aspect='auto', cmap='Greys')
    ax.set_yticks(range(len(missing.columns)))
    ax.set_yticklabels(missing.columns, fontsize=8)
    ax.set_xlabel('Sample index')
    ax.set_title('Missing value matrix (columns on y axis)')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()


def compute_text_length(df, text_builder=None):
    def default_builder(row):
        parts = []
        if isinstance(row.get('texts'), list):
            parts.extend(row.get('texts') or [])
        if isinstance(row.get('symptoms'), list):
            parts.extend(row.get('symptoms') or [])
        if isinstance(row.get('visual_analysis'), list):
            parts.extend(row.get('visual_analysis') or [])
        return '\n'.join(parts).strip()

    builder = text_builder or default_builder
    lengths = []
    texts = []
    for _, row in df.iterrows():
        text = builder(row) if isinstance(row, dict) or hasattr(row, 'to_dict') else ''
        if not isinstance(text, str):
            text = str(text)
        texts.append(text)
        lengths.append(len(text.split()))
    return np.array(lengths), texts


def plot_text_length_distribution_per_class(df, class_col='label', save_path=None):
    lengths, texts = compute_text_length(df)
    df_cp = df.copy()
    df_cp['_text_len'] = lengths
    classes = df_cp[class_col].unique()
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df_cp[df_cp[class_col] == c]['_text_len'].dropna() for c in classes]
    ax.boxplot(data, labels=classes)
    ax.set_ylabel('Text length (words)')
    ax.set_title('Text length distribution per class')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()


def generate_top_ngrams_per_class(df, class_col='label', text_builder=None, ngram_range=(1,2), top_k=30, output_csv=None, output_fig=None):
    lengths, texts = compute_text_length(df, text_builder)
    df_cp = df.copy()
    df_cp['_text'] = texts

    class_list = sorted(df_cp[class_col].unique())
    rows = []
    for cls in class_list:
        texts_cls = df_cp[df_cp[class_col] == cls]['_text'].astype(str).values
        if len(texts_cls) == 0:
            continue
        vect = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=10000)
        X = vect.fit_transform(texts_cls)
        sums = np.array(X.sum(axis=0)).ravel()
        terms = np.array(vect.get_feature_names_out())
        top_idx = sums.argsort()[::-1][:top_k]
        for idx in top_idx:
            rows.append({'class': cls, 'ngram': terms[idx], 'count': int(sums[idx])})

    df_out = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_csv, index=False, encoding='utf-8')

    # simple barplot for top terms across all classes combined (top_k)
    if output_fig:
        top_overall = df_out.groupby('ngram')['count'].sum().sort_values(ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_overall[::-1].plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Top n-grams (all classes)')
        plt.tight_layout()
        Path(output_fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_fig, dpi=300)
        plt.close()

    return df_out


def compute_image_basic_stats(df, image_col='image', sample_limit=200, output_csv=None, save_fig=None):
    widths = []
    heights = []
    brightnesses = []
    paths = []
    for i, row in df.iterrows():
        if i >= sample_limit:
            break
        img_path = row.get(image_col)
        if not img_path:
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            stat = ImageStat.Stat(img)
            # brightness as mean pixel intensity
            brightness = np.mean(stat.mean)
            widths.append(w)
            heights.append(h)
            brightnesses.append(brightness)
            paths.append(img_path)
        except Exception:
            continue

    df_stats = pd.DataFrame({'image_path': paths, 'width': widths, 'height': heights, 'brightness': brightnesses})
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_stats.to_csv(output_csv, index=False, encoding='utf-8')

    if save_fig:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(widths, bins=20, color='skyblue')
        axes[0].set_title('Image widths')
        axes[1].hist(brightnesses, bins=20, color='lightgreen')
        axes[1].set_title('Image brightness (mean)')
        plt.tight_layout()
        Path(save_fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig, dpi=300)
        plt.close()

    return df_stats


def run_basic_eda(metadata_paths, output_dir='outputs/analysis', sample_images=200):
    """metadata_paths: dict with keys like 'train','val','test' and file paths as values"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Focus on train split for most EDA
    train_meta = metadata_paths.get('train')
    if train_meta is None:
        print('No train metadata provided for EDA')
        return

    df = _load_metadata(train_meta)

    # 1. Class distribution
    plot_class_distribution(df, save_path=str(out_dir / 'class_distribution.png'))

    # 2. Metadata correlation (use numeric cols if present)
    numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    # also include some expected numeric names if present
    for k in ['humidity', 'temperature', 'leaf_area_ratio', 'background_ratio', 'lesion_area_ratio', 'annotation_confidence']:
        if k in df.columns and k not in numeric_cols:
            numeric_cols.append(k)
    plot_metadata_correlation(df, numeric_cols, save_path=str(out_dir / 'metadata_correlation.png'))

    # 3. Missing matrix
    plot_missing_matrix(df, save_path=str(out_dir / 'missing_matrix.png'))

    # 4. Text length per class
    plot_text_length_distribution_per_class(df, save_path=str(out_dir / 'text_length_per_class.png'))

    # 5. Top n-grams per class -> csv + overall figure
    generate_top_ngrams_per_class(df, output_csv=str(out_dir / 'ngram_freq.csv'), output_fig=str(out_dir / 'top_ngrams.png'))

    # 6. Image basic stats
    compute_image_basic_stats(df, sample_limit=sample_images, output_csv=str(out_dir / 'image_quality.csv'), save_fig=str(out_dir / 'image_quality.png'))

    print('✓ Basic EDA outputs saved to', out_dir)
    return out_dir

# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    
    # Đường dẫn đến metadata của bạn
    METADATA_PATHS = {
        'train': 'dataset/splits/train.json',
        'val': 'dataset/splits/val.json',
        'test': 'dataset/splits/test.json'
    }
    
    # Thư mục output
    OUTPUT_DIR = 'outputs/analysis'
    
    # Chạy EDA
    print("=" * 60)
    print("RUNNING BASIC EDA")
    print("=" * 60)
    
    result = run_basic_eda(
        metadata_paths=METADATA_PATHS,
        output_dir=OUTPUT_DIR,
        sample_images=200
    )
    
    print("\n✅ EDA Completed Successfully!")
    print(f"📁 Output saved to: {result}")
    
    # Liệt kê các file đã tạo
    print("\n📊 Generated files:")
    output_path = Path(OUTPUT_DIR)
    for file in output_path.glob("*"):
        print(f"  - {file.name}")