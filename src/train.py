import random
from pathlib import Path
import logging
import sys

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.rice_dataset import RiceDiseaseDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from models.multimodal_model import RiceDiseaseMultimodalModel
from trainers.losses import build_loss
from trainers.trainer import Trainer
from evaluation.evaluate import evaluate
from evaluation.plot_metrics import plot_training_curves, save_history_json
from evaluation.confusion_matrix import plot_confusion_matrix
from utils.fs import ensure_dirs
from utils.logger import setup_logger

CONFIG_PATH = "configs/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        raise FileNotFoundError(
            f"Required metadata file not found: {metadata_file.resolve()}"
        )
    return str(metadata_file)


def main():
    # Load config
    config = load_config()
    set_seed(config["training"].get("seed", 42))
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Setup directories
    output_dirs = {
        "checkpoints": Path(config["outputs"]["checkpoints"]),
        "figures": Path(config["outputs"]["figures"]),
        "metrics": Path(config["outputs"]["metrics"]),
        "logs": Path("outputs/logs"),
    }
    
    for dir_path in output_dirs.values():
        ensure_dirs(str(dir_path))
    
    # Setup logger
    log_file = output_dirs["logs"] / "training.log"
    logger = setup_logger(str(log_file))
    logger.info("=" * 60)
    logger.info("RICE DISEASE MULTIMODAL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Config: {CONFIG_PATH}")
    logger.info(f"Device: {device}")
    
    # Validate metadata paths
    train_metadata = validate_metadata_path(config["data"]["train_metadata"])
    val_metadata = validate_metadata_path(config["data"]["val_metadata"])
    test_metadata = validate_metadata_path(config["data"]["test_metadata"])
    
    logger.info(f"Train metadata: {train_metadata}")
    logger.info(f"Val metadata: {val_metadata}")
    logger.info(f"Test metadata: {test_metadata}")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = RiceDiseaseDataset(
        metadata_path=train_metadata,
        max_length=config["data"]["max_length"],
        transform=get_train_transforms(config["data"]["image_size"]),
        deterministic_text=True,  # Fix random issue
    )
    
    val_dataset = RiceDiseaseDataset(
        metadata_path=val_metadata,
        max_length=config["data"]["max_length"],
        transform=get_val_transforms(config["data"]["image_size"]),
        deterministic_text=True,
    )
    
    test_dataset = RiceDiseaseDataset(
        metadata_path=test_metadata,
        max_length=config["data"]["max_length"],
        transform=get_val_transforms(config["data"]["image_size"]),
        deterministic_text=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 2),
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Create model
    logger.info("\nCreating model...")
    model = RiceDiseaseMultimodalModel(
        num_classes=config["model"]["num_classes"],
        fusion_type=config["model"]["fusion"],
    ).to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss, optimizer, scheduler
    criterion = build_loss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )
    
    # Create trainer
    save_path = output_dirs["checkpoints"] / "best_model.pth"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=str(save_path),
        logger=logger,
        patience=config["training"].get("patience", 4),
        grad_clip_norm=config["training"].get("grad_clip_norm", 1.0),
        use_amp=True,
    )
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    history = trainer.fit(config["training"]["epochs"])
    
    # Save training history
    history_path = output_dirs["metrics"] / "training_history.json"
    save_history_json({
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'train_f1s': history['train_f1s'],
        'val_f1s': history['val_f1s'],
        'train_accs': history['train_accs'],
        'val_accs': history['val_accs'],
        'best_f1': history['best_f1']
    }, history_path)
    
    # Plot training curves
    plot_training_curves(history, output_dirs["figures"] / "training_curves.png")
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    class_names = config["dataset"]["classes"]
    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=str(output_dirs["metrics"]),
    )
    
    # Plot confusion matrix
    # Note: evaluate function already saves confusion matrix, but we can also plot additional metrics
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best model saved to: {save_path}")
    logger.info(f"Training history saved to: {history_path}")
    logger.info(f"Figures saved to: {output_dirs['figures']}")
    logger.info(f"Logs saved to: {log_file}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 score: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()