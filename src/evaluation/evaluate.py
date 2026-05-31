import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report

from trainers.metrics import calculate_metrics
from evaluation.confusion_matrix import plot_confusion_matrix
from utils.fs import ensure_dir


@torch.no_grad()
def evaluate(model, dataloader, device, class_names, output_dir="outputs"):
    model.eval()

    y_true = []
    y_pred = []

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images, input_ids, attention_mask)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    metrics = calculate_metrics(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    output_path = Path(output_dir)
    ensure_dir(output_path)
    report_path = output_path / "classification_report.txt"
    metrics_path = output_path / "metrics_summary.json"
    confusion_path = output_path.parent / "figures" / "confusion_matrix.png"
    ensure_dir(confusion_path.parent)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(metrics)
    print(report)

    plot_confusion_matrix(y_true, y_pred, class_names, str(confusion_path))
    return metrics
