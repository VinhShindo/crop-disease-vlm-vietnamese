import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np


def plot_training_curves(history, save_path):
    """Vẽ biểu đồ training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss curve
    axes[0].plot(history['train_losses'], label='Train Loss', marker='o')
    axes[0].plot(history['val_losses'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['train_accs'], label='Train Acc', marker='o')
    axes[1].plot(history['val_accs'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 curve
    axes[2].plot(history['train_f1s'], label='Train F1', marker='o')
    axes[2].plot(history['val_f1s'], label='Val F1', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score Curves')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {save_path}")


def plot_metrics_comparison(train_metrics, val_metrics, save_path):
    """Vẽ biểu đồ so sánh metrics cuối cùng"""
    metrics_names = list(train_metrics.keys())
    train_values = [train_metrics[m] for m in metrics_names]
    val_values = [val_metrics[m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='skyblue')
    bars2 = ax.bar(x + width/2, val_values, width, label='Validation', color='lightcoral')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics comparison saved to {save_path}")


def save_history_json(history, save_path):
    """Lưu history thành JSON"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {save_path}")