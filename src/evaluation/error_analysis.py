import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import numpy as np


@torch.no_grad()
def analyze_errors(model, dataloader, device, class_names, output_dir="outputs"):
    """Phân tích chi tiết các mẫu bị phân loại sai"""
    model.eval()
    
    errors = []  # Lưu thông tin lỗi
    all_predictions = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        metadata = batch.get("metadata")
        
        if metadata is not None and isinstance(metadata, torch.Tensor):
            metadata = metadata.to(device)
        
        outputs = model(images, input_ids, attention_mask, metadata)
        preds = outputs.argmax(dim=1)
        
        # Tìm các mẫu bị sai
        incorrect_mask = preds != labels
        
        if incorrect_mask.any():
            for i in range(len(incorrect_mask)):
                if incorrect_mask[i]:
                    error_info = {
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'image_path': batch["image_path"][i] if "image_path" in batch else "unknown",
                        'true_label': class_names[labels[i].item()],
                        'true_label_id': labels[i].item(),
                        'pred_label': class_names[preds[i].item()],
                        'pred_label_id': preds[i].item(),
                        'raw_text': batch["raw_text"][i] if "raw_text" in batch else "",
                    }
                    
                    # Add metadata if available
                    if "metadata" in batch and batch["metadata"] is not None:
                        if isinstance(batch["metadata"], torch.Tensor):
                            # Metadata is a tensor, can't easily convert to dict
                            error_info['metadata_available'] = True
                        else:
                            error_info['metadata'] = batch["metadata"][i] if i < len(batch["metadata"]) else {}
                    
                    errors.append(error_info)
        
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Tạo DataFrame errors
    df_errors = pd.DataFrame(errors)
    
    # Phân tích confusion pairs
    if len(df_errors) > 0:
        confusion_pairs = df_errors.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
        confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
    else:
        confusion_pairs = pd.DataFrame(columns=['true_label', 'pred_label', 'count'])
    
    # Tạo report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ERROR ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"\nTotal samples analyzed: {len(all_labels)}")
    report_lines.append(f"Total errors: {len(errors)}")
    report_lines.append(f"Error rate: {len(errors)/len(all_labels)*100:.2f}%")
    report_lines.append(f"Accuracy: {(1 - len(errors)/len(all_labels))*100:.2f}%")
    
    report_lines.append("\n" + "-" * 40)
    report_lines.append("CONFUSION PAIRS (True → Predicted):")
    report_lines.append("-" * 40)
    
    if len(confusion_pairs) > 0:
        for _, row in confusion_pairs.iterrows():
            report_lines.append(f"  {row['true_label']} → {row['pred_label']}: {row['count']} errors ({row['count']/len(errors)*100:.1f}%)")
    else:
        report_lines.append("  No errors found!")
    
    # Per-class error analysis
    report_lines.append("\n" + "-" * 40)
    report_lines.append("PER-CLASS ERROR ANALYSIS:")
    report_lines.append("-" * 40)
    
    if len(df_errors) > 0:
        for class_name in class_names:
            class_errors = df_errors[df_errors['true_label'] == class_name]
            total_class_samples = sum(1 for l in all_labels if l == class_names.index(class_name))
            if total_class_samples > 0:
                error_rate = len(class_errors) / total_class_samples * 100
                report_lines.append(f"  {class_name}: {len(class_errors)}/{total_class_samples} errors ({error_rate:.2f}%)")
                
                # Most common confusion for this class
                if len(class_errors) > 0:
                    confused_with = class_errors['pred_label'].value_counts().head(3)
                    confused_str = ", ".join([f"{k}({v})" for k, v in confused_with.items()])
                    report_lines.append(f"    → Confused with: {confused_str}")
    
    # Lưu report
    report_path = output_path / "error_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Lưu chi tiết lỗi ra CSV
    if len(errors) > 0:
        csv_path = output_path / "error_samples.csv"
        df_errors.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ Error samples saved to {csv_path}")
    
    print(f"✓ Error analysis report saved to {report_path}")
    
    return df_errors, confusion_pairs


def plot_error_distribution(df_errors, save_path, class_names):
    """Vẽ biểu đồ phân bố lỗi"""
    if len(df_errors) == 0:
        print("No errors to plot!")
        # Create a figure with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Errors Found!\nModel achieved 100% accuracy on this dataset', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Error Distribution (No Errors)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Error distribution plot saved to {save_path} (no errors)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Errors by true label
    true_counts = df_errors['true_label'].value_counts()
    axes[0].bar(range(len(true_counts)), true_counts.values, color='skyblue', edgecolor='navy')
    axes[0].set_xlabel('True Label', fontsize=11)
    axes[0].set_ylabel('Number of Errors', fontsize=11)
    axes[0].set_title('Errors by True Label', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(true_counts)))
    axes[0].set_xticklabels(true_counts.index, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (label, count) in enumerate(true_counts.items()):
        axes[0].text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Confusion matrix of errors
    confusion = pd.crosstab(df_errors['true_label'], df_errors['pred_label'])
    
    # Ensure all classes are represented
    for class_name in class_names:
        if class_name not in confusion.index:
            confusion.loc[class_name] = 0
        if class_name not in confusion.columns:
            confusion[class_name] = 0
    
    confusion = confusion.reindex(index=class_names, columns=class_names, fill_value=0)
    
    sns.heatmap(confusion, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1], 
                cbar_kws={'label': 'Number of Errors'})
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_title('Error Confusion Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error distribution plot saved to {save_path}")