import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report

from trainers.metrics import calculate_metrics
from evaluation.confusion_matrix import plot_confusion_matrix
from utils.fs import ensure_dir


@torch.no_grad()
def evaluate(model, dataloader, device, class_names, output_dir="outputs/metrics"):
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

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*50)
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(y_true, y_pred, class_names, str(confusion_path))
    return metrics

def evaluate_with_analysis(model, dataloader, device, class_names, output_dir="outputs"):
    """Evaluate with detailed analysis"""
    
    # Import các functions mới
    from evaluation.tsne_visualization import extract_embeddings, plot_tsne_embeddings, plot_embedding_separation
    from evaluation.error_analysis import analyze_errors, plot_error_distribution
    
    output_path = Path(output_dir)
    
    # 1. Standard evaluation
    metrics = evaluate(model, dataloader, device, class_names, output_dir)
    
    # 2. Extract embeddings for t-SNE
    print("\nExtracting embeddings for t-SNE...")
    embeddings_dict = extract_embeddings(model, dataloader, device, output_dir)
    
    # 3. Plot t-SNE
    tsne_path = output_path.parent / "figures" / "tsne_visualization.png"
    plot_tsne_embeddings(embeddings_dict, tsne_path, class_names)
    
    # 4. Plot embedding separation
    separation_path = output_path.parent / "figures" / "embedding_separation.png"
    plot_embedding_separation(embeddings_dict, separation_path, class_names)
    
    # 5. Error analysis
    df_errors, confusion_pairs = analyze_errors(model, dataloader, device, class_names, output_dir)
    
    # 6. Plot error distribution
    if len(df_errors) > 0:
        error_plot_path = output_path.parent / "figures" / "error_distribution.png"
        plot_error_distribution(df_errors, error_plot_path, class_names)
    
    return metrics, embeddings_dict, df_errors