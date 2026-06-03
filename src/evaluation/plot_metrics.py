import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
import pandas as pd


def plot_training_curves(history, save_path):
    """Vẽ biểu đồ training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve
    axes[0, 0].plot(history['train_losses'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(history['val_losses'], label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[0, 1].plot(history['train_accs'], label='Train Acc', marker='o', linewidth=2)
    axes[0, 1].plot(history['val_accs'], label='Val Acc', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 curve
    axes[1, 0].plot(history['train_f1s'], label='Train F1', marker='o', linewidth=2)
    axes[1, 0].plot(history['val_f1s'], label='Val F1', marker='s', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score Curves', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gap between train and val
    gap = np.array(history['train_accs']) - np.array(history['val_accs'])
    axes[1, 1].fill_between(range(len(gap)), 0, gap, alpha=0.3, color='red', label='Overfitting Gap')
    axes[1, 1].plot(gap, marker='o', color='red', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train-Val Gap')
    axes[1, 1].set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


def plot_confusion_matrix_advanced(cm, class_names, save_path):
    """Vẽ confusion matrix với percentages"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annotations = np.empty_like(cm, dtype='U20')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix with Percentages', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Advanced confusion matrix saved to {save_path}")


def plot_class_performance(class_metrics, save_path):
    """Vẽ biểu đồ performance theo từng class"""
    classes = list(class_metrics.keys())
    precision = [class_metrics[c]['precision'] for c in classes]
    recall = [class_metrics[c]['recall'] for c in classes]
    f1 = [class_metrics[c]['f1'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Class performance chart saved to {save_path}")


def plot_metadata_impact(history_with_meta, history_without_meta, save_path):
    """So sánh performance có và không có metadata"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    axes[0].plot(history_with_meta['val_accs'], label='With Metadata', marker='o', linewidth=2)
    axes[0].plot(history_without_meta['val_accs'], label='Without Metadata', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Metadata Impact on Accuracy', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 comparison
    axes[1].plot(history_with_meta['val_f1s'], label='With Metadata', marker='o', linewidth=2)
    axes[1].plot(history_without_meta['val_f1s'], label='Without Metadata', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation F1 Score')
    axes[1].set_title('Metadata Impact on F1 Score', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metadata impact chart saved to {save_path}")


def save_history_json(history, save_path):
    """Lưu history thành JSON"""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    history_clean = {k: convert(v) for k, v in history.items()}
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(history_clean, f, indent=2, ensure_ascii=False)
    print(f"✓ Training history saved to {save_path}")