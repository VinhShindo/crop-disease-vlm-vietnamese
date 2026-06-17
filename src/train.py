import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from datasets.rice_dataset import RiceDiseaseDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from models.multimodal_model import RiceDiseaseMultimodalModel
from trainers.losses import build_loss
from evaluation.evaluate import evaluate
from evaluation.plot_metrics import (
    plot_training_curves, save_history_json
)
from evaluation.generate_summary_csv import generate_training_summary, generate_config_summary
from utils.fs import ensure_dirs
from utils.logger import setup_logger

from evaluation.tsne_visualization import (
    extract_embeddings,
    plot_tsne_embeddings,
    plot_embedding_separation,
    plot_tsne_comparison
)
from evaluation.visual_analysis import (
    compute_probs_and_labels,
    plot_roc_pr_curves,
    plot_calibration_curve,
    plot_confidence_histogram,
    plot_misclassification_gallery,
)

CONFIG_PATH = "configs/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if "training" in config:
        numeric_params = ["lr", "weight_decay", "grad_clip_norm", "batch_size", "epochs", "patience", 
                         "warmup_epochs", "min_lr", "label_smoothing"]
        for param in numeric_params:
            if param in config["training"] and isinstance(config["training"][param], str):
                if param in ["batch_size", "epochs", "patience", "warmup_epochs"]:
                    config["training"][param] = int(config["training"][param])
                else:
                    config["training"][param] = float(config["training"][param])
    
    if "leaf_attention" not in config:
        config["leaf_attention"] = {
            "use_leaf_mask": True,
            "attention_weight": 0.15,
            "min_coverage_threshold": 0.05,
        }
    
    return config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_metadata_path(path: str):
    metadata_file = Path(path)
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {metadata_file.resolve()}")
    return str(metadata_file)


class LeafAttentionLoss(nn.Module):
    """Loss với attention dựa trên leaf coverage và label smoothing"""
    def __init__(self, base_loss, attention_weight=0.15, min_coverage=0.05):
        super().__init__()
        self.base_loss = base_loss
        self.attention_weight = attention_weight
        self.min_coverage = min_coverage
    
    def forward(self, predictions, targets, leaf_coverage=None):
        loss = self.base_loss(predictions, targets)
        
        if leaf_coverage is not None and self.attention_weight > 0:
            attention_mask = (leaf_coverage > self.min_coverage).float()
            attention_weights = 1.0 + self.attention_weight * \
                               (leaf_coverage - self.min_coverage) / (1.0 - self.min_coverage)
            attention_weights = attention_weights * attention_mask
            
            if attention_weights.sum() > 0:
                loss = loss * attention_weights.mean()
        
        return loss


class MaskGuidedTrainer:
    """Trainer hỗ trợ leaf mask với progress bar và early stopping cải thiện"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler,
                 device, save_path, logger, patience=6, grad_clip_norm=0.5, use_amp=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.logger = logger
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp
        
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_losses': [], 'val_losses': [], 
            'train_f1s': [], 'val_f1s': [],
            'train_accs': [], 'val_accs': [],
            'learning_rates': []
        }
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training", unit="batch", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask_text = batch['attention_mask_text'].to(self.device)
            labels = batch['label'].to(self.device)
            metadata = batch.get('metadata')
            if metadata is not None:
                metadata = metadata.to(self.device)
            
            # Lấy leaf mask và coverage
            leaf_mask = batch.get('leaf_mask')
            if leaf_mask is not None:
                leaf_mask = leaf_mask.to(self.device)
            
            leaf_coverage = batch.get('leaf_coverage')
            if leaf_coverage is not None:
                leaf_coverage = leaf_coverage.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        images, input_ids, attention_mask_text, metadata,
                        leaf_mask=leaf_mask
                    )
                    loss = self.criterion(outputs, labels, leaf_coverage=leaf_coverage)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    images, input_ids, attention_mask_text, metadata,
                    leaf_mask=leaf_mask
                )
                loss = self.criterion(outputs, labels, leaf_coverage=leaf_coverage)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if (batch_idx + 1) % 50 == 0:
                self.logger.info(f"  Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return total_loss / len(self.train_loader), acc, f1
    
    def evaluate(self, loader, name="Validation"):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        pbar = tqdm(loader, desc=name, unit="batch", leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask_text = batch['attention_mask_text'].to(self.device)
                labels = batch['label'].to(self.device)
                metadata = batch.get('metadata')
                if metadata is not None:
                    metadata = metadata.to(self.device)
                
                leaf_mask = batch.get('leaf_mask')
                if leaf_mask is not None:
                    leaf_mask = leaf_mask.to(self.device)
                
                leaf_coverage = batch.get('leaf_coverage')
                if leaf_coverage is not None:
                    leaf_coverage = leaf_coverage.to(self.device)
                
                outputs = self.model(
                    images, input_ids, attention_mask_text, metadata,
                    leaf_mask=leaf_mask
                )
                loss = self.criterion(outputs, labels, leaf_coverage=leaf_coverage)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return total_loss / len(loader), acc, f1
    
    def fit(self, epochs):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STARTING TRAINING WITH LEAF MASK ATTENTION")
        self.logger.info(f"  Batch size: {self.train_loader.batch_size}")
        self.logger.info(f"  Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
        self.logger.info(f"  Patience: {self.patience}")
        self.logger.info(f"  Grad clip norm: {self.grad_clip_norm}")
        self.logger.info("=" * 60)
        
        epoch_pbar = tqdm(range(epochs), desc="Epochs", unit="epoch")
        
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.evaluate(self.val_loader, "Validating")
            
            # Update history
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['train_f1s'].append(train_f1)
            self.history['val_f1s'].append(val_f1)
            self.history['train_accs'].append(train_acc)
            self.history['val_accs'].append(val_acc)
            
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.history['learning_rates'].append(current_lr)
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
            
            # Log results
            self.logger.info(f"\n📊 Epoch {epoch+1}/{epochs}")
            self.logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%) | Train F1: {train_f1:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%) | Val F1: {val_f1:.4f}")
            self.logger.info(f"  LR: {current_lr:.6f}")
            
            # Early stopping check dựa trên F1 score (chính xác hơn)
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_epoch = epoch + 1
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                self.logger.info(f"  ✓ Best model saved! (F1: {self.best_f1:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.logger.info(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    self.logger.info(f"\n⏹ Early stopping triggered at epoch {epoch+1}")
                    self.logger.info(f"  Best model was at epoch {self.best_epoch} with F1: {self.best_f1:.4f}")
                    break
            
            # Cảnh báo overfit
            if val_loss > train_loss * 1.2 and epoch > 5:
                self.logger.warning(f"  ⚠️ Possible overfitting: Val Loss ({val_loss:.4f}) > Train Loss ({train_loss:.4f})")
            
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_f1': f'{val_f1:.4f}'
            })
        
        self.history['best_f1'] = self.best_f1
        self.history['best_epoch'] = self.best_epoch
        self.history['best_val_loss'] = self.best_val_loss
        return self.history


def main():
    config = load_config()
    set_seed(config["training"].get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # Setup directories
    output_dirs = {
        "checkpoints": Path(config["outputs"]["checkpoints"]),
        "figures": Path(config["outputs"]["figures"]),
        "metrics": Path(config["outputs"]["metrics"]),
        "logs": Path(config["outputs"]["logs"]),
        "csv": Path("outputs/csv"),
        "gradcam": Path("outputs/figures/gradcam"),
    }
    
    for dir_path in output_dirs.values():
        ensure_dirs(str(dir_path))
    
    # Setup logger
    log_file = output_dirs["logs"] / "training.log"
    logger = setup_logger(str(log_file))
    logger.info("=" * 80)
    logger.info("🌾 RICE DISEASE MULTIMODAL TRAINING (WITH LEAF MASK ATTENTION)")
    logger.info("=" * 80)
    logger.info(f"📁 Config: {CONFIG_PATH}")
    logger.info(f"💻 Device: {device}")
    
    # Validate metadata paths
    train_metadata = validate_metadata_path(config["data"]["train_metadata"])
    val_metadata = validate_metadata_path(config["data"]["val_metadata"])
    test_metadata = validate_metadata_path(config["data"]["test_metadata"])
    
    logger.info(f"\n📂 Data paths:")
    logger.info(f"  Train: {train_metadata}")
    logger.info(f"  Val: {val_metadata}")
    logger.info(f"  Test: {test_metadata}")
    
    logger.info("\n📦 Creating datasets with leaf mask...")
    
    leaf_config = config.get("leaf_attention", {})
    use_leaf_mask = leaf_config.get("use_leaf_mask", True)
    
    logger.info(f"  Use leaf mask: {use_leaf_mask}")
    
    train_dataset = RiceDiseaseDataset(
        metadata_path=train_metadata,
        tokenizer_name="vinai/phobert-base",
        max_length=config["data"]["max_length"],
        transform=get_train_transforms(config["data"]["image_size"]),
        deterministic_text=True,
        use_metadata=True,
        extract_leaf_metrics=True,
        image_size_for_extraction=512,
        use_leaf_mask=use_leaf_mask,
    )
    
    val_dataset = RiceDiseaseDataset(
        metadata_path=val_metadata,
        tokenizer_name="vinai/phobert-base",
        max_length=config["data"]["max_length"],
        transform=get_val_transforms(config["data"]["image_size"]),
        deterministic_text=True,
        use_metadata=True,
        extract_leaf_metrics=True,
        image_size_for_extraction=512,
        use_leaf_mask=use_leaf_mask,
    )
    
    test_dataset = RiceDiseaseDataset(
        metadata_path=test_metadata,
        tokenizer_name="vinai/phobert-base",
        max_length=config["data"]["max_length"],
        transform=get_val_transforms(config["data"]["image_size"]),
        deterministic_text=True,
        use_metadata=True,
        extract_leaf_metrics=True,
        image_size_for_extraction=512,
        use_leaf_mask=use_leaf_mask,
    )
    
    logger.info(f"\n📊 Dataset statistics:")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Log leaf metrics statistics
    if hasattr(train_dataset, 'leaf_metrics_cache') and train_dataset.leaf_metrics_cache:
        coverages = [m['coverage'] for m in train_dataset.leaf_metrics_cache.values() if m]
        aspect_ratios = [m['aspect_ratio'] for m in train_dataset.leaf_metrics_cache.values() if m]
        has_mask = [1 if m.get('hull_mask') is not None else 0 for m in train_dataset.leaf_metrics_cache.values()]
        
        logger.info(f"\n🍃 Leaf Metrics Statistics:")
        logger.info(f"  Coverage - Mean: {np.mean(coverages):.3f}, Std: {np.std(coverages):.3f}")
        logger.info(f"  Coverage - Min: {np.min(coverages):.3f}, Max: {np.max(coverages):.3f}")
        logger.info(f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Std: {np.std(aspect_ratios):.2f}")
        logger.info(f"  Valid hull masks: {sum(has_mask)}/{len(train_dataset)} ({sum(has_mask)/len(train_dataset)*100:.1f}%)")
    
    # Create dataloaders với prefetch
    num_workers = min(config["data"].get("num_workers", 4), 8)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=min(num_workers, 2), 
        pin_memory=True
    )
    
    logger.info("\n🏗 Creating model with leaf mask attention...")
    model = RiceDiseaseMultimodalModel(
        num_classes=config["model"]["num_classes"],
        fusion_type=config["model"]["fusion"],
        use_metadata=True,
        use_leaf_mask=use_leaf_mask,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model fusion type: {config['model']['fusion']}")
    
    try:
        generate_config_summary(config, output_dirs["csv"] / "config_summary.csv")
        logger.info(f"✓ Config summary saved to {output_dirs['csv'] / 'config_summary.csv'}")
    except Exception as e:
        logger.warning(f"Could not generate config summary: {e}")
    
    # Create loss với label smoothing
    label_smoothing = config["training"].get("label_smoothing", 0.05)
    base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    logger.info(f"\n📉 Loss function:")
    logger.info(f"  Base loss: CrossEntropyLoss with label_smoothing={label_smoothing}")
    logger.info(f"  Attention weight: {leaf_config.get('attention_weight', 0.15)}")
    logger.info(f"  Min coverage threshold: {leaf_config.get('min_coverage_threshold', 0.05)}")
    
    criterion = LeafAttentionLoss(
        base_loss=base_criterion,
        attention_weight=leaf_config.get("attention_weight", 0.15),
        min_coverage=leaf_config.get("min_coverage_threshold", 0.05)
    )
    
    # Optimizer
    weight_decay = float(config["training"].get("weight_decay", 5e-4))
    optimizer = AdamW(
        model.parameters(), 
        lr=float(config["training"]["lr"]),
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler với warmup
    epochs = int(config["training"]["epochs"])
    warmup_epochs = config["training"].get("warmup_epochs", 3)
    min_lr = config["training"].get("min_lr", 1e-6)
    
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        logger.info(f"\n⚙️ Scheduler: Warmup({warmup_epochs} epochs) + CosineAnnealing (min_lr={min_lr})")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        logger.info(f"\n⚙️ Scheduler: CosineAnnealingLR (min_lr={min_lr})")
    
    logger.info(f"  Learning rate: {config['training']['lr']}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    
    save_path = output_dirs["checkpoints"] / "best_model.pth"
    
    # Trainer với các tham số mới
    trainer = MaskGuidedTrainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler,
        device=device, 
        save_path=str(save_path), 
        logger=logger,
        patience=int(config["training"].get("patience", 6)),
        grad_clip_norm=float(config["training"].get("grad_clip_norm", 0.5)),
        use_amp=config["training"].get("use_amp", True),
    )
    
    # Train
    history = trainer.fit(epochs)
    
    # Save training history
    history_path = output_dirs["metrics"] / "training_history.json"
    save_history_json(history, history_path)
    logger.info(f"\n✓ Training history saved to {history_path}")
    
    # Plot training curves
    plot_training_curves(history, output_dirs["figures"] / "training_curves.png")
    logger.info(f"✓ Training curves saved to {output_dirs['figures'] / 'training_curves.png'}")
    
    # Evaluate on test set
    logger.info("\n🔍 EVALUATING ON TEST SET")
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    class_names = config["dataset"]["classes"]
    test_metrics = evaluate(
        model=model, 
        dataloader=test_loader, 
        device=device,
        class_names=class_names, 
        output_dir=str(output_dirs["metrics"])
    )
    
    logger.info(f"\n📊 Test Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    
    # Additional visualizations (giữ nguyên)
    try:
        logger.info("\n📈 Generating additional visualizations...")
        probs_labels = compute_probs_and_labels(model, test_loader, device)
        vis_dir = output_dirs["figures"]
        
        plot_roc_pr_curves(probs_labels['probs'], probs_labels['labels'], 
                          class_names, str(vis_dir / 'roc_pr_curves.png'))
        logger.info("  ✓ ROC/PR curves saved")
        
        for i in range(min(2, len(class_names))):
            plot_calibration_curve(probs_labels['probs'], probs_labels['labels'], i, 
                                  str(vis_dir / f'calibration_class_{i}.png'))
        logger.info("  ✓ Calibration curves saved")
        
        plot_confidence_histogram(probs_labels['probs'], probs_labels['preds'], probs_labels['labels'],
                                  str(vis_dir / 'confidence_histogram.png'))
        logger.info("  ✓ Confidence histogram saved")
        
        plot_misclassification_gallery(probs_labels['probs'], probs_labels['preds'], probs_labels['labels'],
                                       probs_labels['image_paths'], class_names,
                                       str(vis_dir / 'misclassification_gallery.png'))
        logger.info("  ✓ Misclassification gallery saved")
        
        logger.info("✓ Additional visualizations completed")
    except Exception as e:
        logger.warning(f"Could not generate additional visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # t-SNE visualization
    logger.info("\n🎨 Generating t-SNE visualizations...")
    try:
        subset_size = min(300, len(test_dataset))
        subset_indices = list(range(subset_size))
        subset_dataset = Subset(test_dataset, subset_indices)
        subset_loader = DataLoader(subset_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        logger.info(f"  Extracting embeddings from {subset_size} samples...")
        embeddings_dict = extract_embeddings(model, subset_loader, device, str(output_dirs["metrics"]))
        
        tsne_path = output_dirs["figures"] / "tsne_visualization.png"
        plot_tsne_embeddings(embeddings_dict, tsne_path, class_names)
        logger.info(f"  ✓ t-SNE visualization saved")
        
        separation_path = output_dirs["figures"] / "embedding_separation.png"
        plot_embedding_separation(embeddings_dict, separation_path, class_names)
        logger.info(f"  ✓ Embedding separation saved")
        
        comparison_path = output_dirs["figures"] / "tsne_comparison.png"
        plot_tsne_comparison(embeddings_dict, comparison_path, class_names)
        logger.info(f"  ✓ t-SNE comparison saved")
        
        logger.info("✓ t-SNE visualizations completed")
    except Exception as e:
        logger.warning(f"t-SNE visualization failed: {e}")
    
    # Error analysis
    logger.info("\n🔍 Performing error analysis...")
    try:
        from evaluation.error_analysis import analyze_errors, plot_error_distribution
        
        df_errors, confusion_pairs = analyze_errors(
            model, test_loader, device, class_names, str(output_dirs["metrics"])
        )
        
        if len(df_errors) > 0:
            error_plot_path = output_dirs["figures"] / "error_distribution.png"
            plot_error_distribution(df_errors, error_plot_path, class_names)
            
            logger.info(f"  Found {len(df_errors)} error samples ({len(df_errors)/len(test_dataset)*100:.2f}%)")
            logger.info("  Confusion pairs:")
            for _, row in confusion_pairs.iterrows():
                logger.info(f"    {row['true_label']} → {row['pred_label']}: {row['count']} errors")
        else:
            logger.info("  ✓ No errors found! Model achieved perfect predictions")
    except Exception as e:
        logger.warning(f"Error analysis failed: {e}")
    
    # GradCAM visualizations
    logger.info("\n🔥 Generating GradCAM visualizations...")
    try:
        from evaluation.xai_analysis import generate_gradcam_visualization
        
        generate_gradcam_visualization(
            model, 
            test_loader, 
            device, 
            class_names, 
            str(output_dirs["figures"]), 
            max_samples=10
        )
        logger.info("  ✓ GradCAM visualizations saved")
    except Exception as e:
        logger.warning(f"GradCAM visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate final CSV summary
    report_path = output_dirs["metrics"] / "classification_report.txt"
    class_report = None
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            class_report = f.read()
    
    try:
        generate_training_summary(
            config=config,
            history=history,
            test_metrics=test_metrics,
            class_report=class_report,
            output_path=output_dirs["csv"] / "training_summary.csv"
        )
        logger.info(f"✓ Training summary saved")
    except Exception as e:
        logger.warning(f"Could not generate training summary: {e}")
    
    # Final report
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"📁 Best model saved to: {save_path}")
    logger.info(f"📁 Training history: {history_path}")
    logger.info(f"📁 Figures: {output_dirs['figures']}")
    logger.info(f"📁 CSV summaries: {output_dirs['csv']}")
    logger.info(f"📁 Logs: {log_file}")
    logger.info("")
    logger.info(f"🏆 Best Model: Epoch {history.get('best_epoch', 'N/A')} with F1: {history.get('best_f1', 0):.4f}")
    logger.info("")
    logger.info("📊 FINAL TEST RESULTS:")
    logger.info(f"   Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"   Test F1 Score: {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    logger.info(f"   Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"   Test Recall: {test_metrics['recall']:.4f}")
    logger.info("=" * 80)
    
    # Print to console as well
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"🏆 Best Model: Epoch {history.get('best_epoch', 'N/A')} with F1: {history.get('best_f1', 0):.4f}")
    print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"📊 Test F1 Score: {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()