import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_recall_fscore_support
)
from sklearn.calibration import calibration_curve


@torch.no_grad()
def compute_probs_and_labels(model, dataloader, device):
    model.eval()
    probs_list = []
    labels_list = []
    preds_list = []
    paths = []

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(images, input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        probs_list.append(probs.cpu().numpy())
        preds_list.append(preds.cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy())
        paths.extend(batch.get("image_path", [None] * outputs.shape[0]))

    probs_all = np.vstack(probs_list)
    preds_all = np.concatenate(preds_list)
    labels_all = np.concatenate(labels_list)

    return {
        "probs": probs_all,
        "preds": preds_all,
        "labels": labels_all,
        "image_paths": paths,
    }


def plot_roc_pr_curves(probs, labels, class_names, save_path):
    n_classes = len(class_names)
    y_true = label_binarize(labels, classes=list(range(n_classes)))

    plt.figure(figsize=(12, 5))

    # ROC curves (left)
    plt.subplot(1, 2, 1)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(fontsize=8)

    # PR curves (right)
    plt.subplot(1, 2, 2)
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], probs[:, i])
        ap = average_precision_score(y_true[:, i], probs[:, i])
        plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP={ap:.2f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC/PR curves saved to {save_path}")


def plot_calibration_curve(probs, labels, class_idx, save_path, n_bins=10):
    # Plot calibration for one class index
    prob_pos = probs[:, class_idx]
    true_bin = (labels == class_idx).astype(int)
    frac_pos, mean_pred = calibration_curve(true_bin, prob_pos, n_bins=n_bins)

    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, 's-', label=f'Class: {class_idx}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration curve (class={class_idx})')
    plt.legend()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Calibration saved to {save_path}")


def plot_confidence_histogram(probs, preds, labels, save_path):
    confidences = probs.max(axis=1)
    correct_mask = preds == labels
    plt.figure(figsize=(8, 4))
    plt.hist([confidences[correct_mask], confidences[~correct_mask]], bins=30,
             label=['Correct', 'Incorrect'], color=['green', 'red'], alpha=0.7)
    plt.xlabel('Predicted probability (confidence)')
    plt.ylabel('Count')
    plt.title('Confidence distribution: correct vs incorrect')
    plt.legend()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confidence histogram saved to {save_path}")


def plot_misclassification_gallery(probs, preds, labels, image_paths, class_names, save_path, max_images=16):
    # Collect misclassified indices
    mis_idx = np.where(preds != labels)[0]
    if len(mis_idx) == 0:
        print("No misclassified samples to show")
        return

    selected = mis_idx[:max_images]
    n = len(selected)
    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(selected):
        img_path = image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # fallback to blank image
            img = Image.new('RGB', (224, 224), (255, 255, 255))

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        pred_label = class_names[int(preds[idx])]
        true_label = class_names[int(labels[idx])]
        conf = probs[idx, int(preds[idx])]
        plt.title(f"T: {true_label}\nP: {pred_label} ({conf:.2f})", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Misclassification gallery saved to {save_path}")
