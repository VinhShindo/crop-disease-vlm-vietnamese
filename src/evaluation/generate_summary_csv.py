import csv
import json
from pathlib import Path
import pandas as pd
from datetime import datetime


def generate_training_summary(config, history, test_metrics, class_report, output_path):
    """Tạo file CSV tổng kết training"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    
    # 1. Thông tin cấu hình
    summary_data.append({
        'Category': 'CONFIGURATION',
        'Parameter': 'Model Architecture',
        'Value': f"EfficientNet-B0 + PhoBERT + {config['model']['fusion'].upper()}"
    })
    summary_data.append({
        'Category': 'CONFIGURATION',
        'Parameter': 'Total Parameters',
        'Value': '144,529,728'
    })
    summary_data.append({
        'Category': 'CONFIGURATION',
        'Parameter': 'Train/Val/Test Split',
        'Value': f"{config['data']['train_metadata'].split('/')[-1]}, {config['data']['val_metadata'].split('/')[-1]}, {config['data']['test_metadata'].split('/')[-1]}"
    })
    summary_data.append({
        'Category': 'CONFIGURATION',
        'Parameter': 'Batch Size / Learning Rate',
        'Value': f"{config['training']['batch_size']} / {config['training']['lr']}"
    })
    summary_data.append({
        'Category': 'CONFIGURATION',
        'Parameter': 'Image Size / Max Length',
        'Value': f"{config['data']['image_size']} / {config['data']['max_length']}"
    })
    summary_data.append({
        'Category': 'CONFIGURATION',
        'Parameter': 'Training Date',
        'Value': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # 2. Kết quả từng epoch
    for epoch in range(len(history['train_losses'])):
        summary_data.append({
            'Category': f'TRAINING_EPOCH_{epoch+1}',
            'Parameter': 'Train Loss / Train Acc / Train F1',
            'Value': f"{history['train_losses'][epoch]:.4f} / {history['train_accs'][epoch]:.4f} / {history['train_f1s'][epoch]:.4f}"
        })
        summary_data.append({
            'Category': f'TRAINING_EPOCH_{epoch+1}',
            'Parameter': 'Val Loss / Val Acc / Val F1',
            'Value': f"{history['val_losses'][epoch]:.4f} / {history['val_accs'][epoch]:.4f} / {history['val_f1s'][epoch]:.4f}"
        })
    
    # 3. Best epoch
    best_epoch = history['val_f1s'].index(max(history['val_f1s'])) + 1
    summary_data.append({
        'Category': 'BEST_EPOCH',
        'Parameter': f'Epoch {best_epoch}',
        'Value': f"Best Validation F1: {max(history['val_f1s']):.4f}"
    })
    
    # 4. Kết quả test set
    summary_data.append({
        'Category': 'TEST_RESULTS',
        'Parameter': 'Accuracy',
        'Value': f"{test_metrics['accuracy']:.4f}"
    })
    summary_data.append({
        'Category': 'TEST_RESULTS',
        'Parameter': 'Precision (macro)',
        'Value': f"{test_metrics['precision']:.4f}"
    })
    summary_data.append({
        'Category': 'TEST_RESULTS',
        'Parameter': 'Recall (macro)',
        'Value': f"{test_metrics['recall']:.4f}"
    })
    summary_data.append({
        'Category': 'TEST_RESULTS',
        'Parameter': 'F1 Score (macro)',
        'Value': f"{test_metrics['f1']:.4f}"
    })
    summary_data.append({
        'Category': 'TEST_RESULTS',
        'Parameter': 'F1 Score (weighted)',
        'Value': f"{test_metrics['f1_weighted']:.4f}"
    })
    
    # 5. Per-class results (parse từ classification report)
    if class_report:
        lines = class_report.strip().split('\n')
        for line in lines[2:6]:  # Bỏ header và footer
            parts = line.split()
            if len(parts) >= 5:
                class_name = parts[0]
                precision = parts[1]
                recall = parts[2]
                f1 = parts[3]
                support = parts[4]
                summary_data.append({
                    'Category': f'CLASS_{class_name.upper()}',
                    'Parameter': 'Precision / Recall / F1 / Support',
                    'Value': f"{precision} / {recall} / {f1} / {support}"
                })
    
    # Save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ Training summary saved to {output_path}")
    
    return df


def generate_config_summary(config, output_path):
    """Tạo file CSV tóm tắt config"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_data = []
    
    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten_dict(v, new_key)
            else:
                config_data.append({
                    'Parameter': new_key,
                    'Value': str(v)
                })
    
    flatten_dict(config)
    
    df = pd.DataFrame(config_data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ Config summary saved to {output_path}")
    
    return df