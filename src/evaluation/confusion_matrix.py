import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import confusion_matrix

from utils.fs import ensure_dir


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
