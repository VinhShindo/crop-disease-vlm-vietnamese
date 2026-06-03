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
import yaml

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.rice_dataset import RiceDiseaseDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from models.multimodal_model import RiceDiseaseMultimodalModel
from trainers.losses import build_loss
from trainers.trainer import Trainer
from evaluation.evaluate import evaluate
from evaluation.plot_metrics import (
    plot_training_curves, save_history_json
)
from evaluation.generate_summary_csv import generate_training_summary, generate_config_summary
from utils.fs import ensure_dirs
from utils.logger import setup_logger

# Import from tsne_visualization
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
        numeric_params = ["lr", "weight_decay", "grad_clip_norm", "batch_size", "epochs", "patience"]
        for param in numeric_params:
            if param in config["training"] and isinstance(config["training"][param], str):
                if param in ["batch_size", "epochs", "patience"]:
                    config["training"][param] = int(config["training"][param])
                else:
                    config["training"][param] = float(config["training"][param])
    
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
        "logs": Path(config["outputs"]["logs"]),
        "csv": Path("outputs/csv"),
        "gradcam": Path("outputs/figures/gradcam"),
    }
    
    for dir_path in output_dirs.values():
        ensure_dirs(str(dir_path))
    
    # Setup logger
    log_file = output_dirs["logs"] / "training.log"
    logger = setup_logger(str(log_file))
    logger.info("=" * 60)
    logger.info("RICE DISEASE MULTIMODAL TRAINING (WITH METADATA)")
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
    logger.info("\nCreating datasets with metadata...")
    train_dataset = RiceDiseaseDataset(
        metadata_path=train_metadata,
        tokenizer_name="vinai/phobert-base",
        max_length=config["data"]["max_length"],
        transform=get_train_transforms(config["data"]["image_size"]),
        deterministic_text=True,
        use_metadata=True,
    )
    
    val_dataset = RiceDiseaseDataset(
        metadata_path=val_metadata,
        tokenizer_name="vinai/phobert-base",
        max_length=config["data"]["max_length"],
        transform=get_val_transforms(config["data"]["image_size"]),
        deterministic_text=True,
        use_metadata=True,
    )
    
    test_dataset = RiceDiseaseDataset(
        metadata_path=test_metadata,
        tokenizer_name="vinai/phobert-base",
        max_length=config["data"]["max_length"],
        transform=get_val_transforms(config["data"]["image_size"]),
        deterministic_text=True,
        use_metadata=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    num_workers = min(config["data"].get("num_workers", 4), 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=min(num_workers, 2),
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Create model
    logger.info("\nCreating model with metadata integration...")
    model = RiceDiseaseMultimodalModel(
        num_classes=config["model"]["num_classes"],
        fusion_type=config["model"]["fusion"],
        use_metadata=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Save config summary
    try:
        generate_config_summary(config, output_dirs["csv"] / "config_summary.csv")
    except Exception as e:
        logger.warning(f"Could not generate config summary: {e}")
    
    # Create trainer
    criterion = build_loss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
    )
    
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
        patience=int(config["training"].get("patience", 4)),
        grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
        use_amp=True,
    )
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    history = trainer.fit(int(config["training"]["epochs"]))
    
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
    
    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    class_names = config["dataset"]["classes"]
    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=str(output_dirs["metrics"]),
    )

    # Additional visualizations (ROC/PR, calibration, confidence, misclassification)
    try:
        logger.info("Generating additional visualizations...")
        probs_labels = compute_probs_and_labels(model, test_loader, device)
        vis_dir = output_dirs["figures"]
        plot_roc_pr_curves(probs_labels['probs'], probs_labels['labels'], class_names, str(vis_dir / 'roc_pr_curves.png'))
        # Calibration per class (save first 2 classes as example)
        for i in range(min(2, len(class_names))):
            plot_calibration_curve(probs_labels['probs'], probs_labels['labels'], i, str(vis_dir / f'calibration_class_{i}.png'))
        plot_confidence_histogram(probs_labels['probs'], probs_labels['preds'], probs_labels['labels'], str(vis_dir / 'confidence_histogram.png'))
        plot_misclassification_gallery(probs_labels['probs'], probs_labels['preds'], probs_labels['labels'], probs_labels['image_paths'], class_names, str(vis_dir / 'misclassification_gallery.png'))
        logger.info('✓ Additional visualizations completed')
    except Exception as e:
        logger.warning(f'Could not generate additional visualizations: {e}')
        import traceback
        traceback.print_exc()

    # ========== t-SNE VISUALIZATION ==========
    print("\n" + "=" * 60)
    print("GENERATING t-SNE VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Use subset for faster processing (300 samples is enough for t-SNE)
        subset_size = min(300, len(test_dataset))
        subset_indices = list(range(subset_size))
        subset_dataset = Subset(test_dataset, subset_indices)
        subset_loader = DataLoader(subset_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        logger.info(f"Extracting embeddings from {subset_size} samples...")
        embeddings_dict = extract_embeddings(model, subset_loader, device, str(output_dirs["metrics"]))
        
        # Plot t-SNE
        tsne_path = output_dirs["figures"] / "tsne_visualization.png"
        plot_tsne_embeddings(embeddings_dict, tsne_path, class_names)
        
        # Plot embedding separation
        separation_path = output_dirs["figures"] / "embedding_separation.png"
        plot_embedding_separation(embeddings_dict, separation_path, class_names)
        
        # Plot t-SNE comparison
        comparison_path = output_dirs["figures"] / "tsne_comparison.png"
        plot_tsne_comparison(embeddings_dict, comparison_path, class_names)
        
        logger.info("✓ t-SNE visualizations completed")
        
    except ImportError as e:
        logger.warning(f"t-SNE modules not available: {e}")
        print("Skipping t-SNE visualization...")
    except Exception as e:
        logger.error(f"Error in t-SNE visualization: {e}")
        print(f"Warning: Could not generate t-SNE plots: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Error analysis
    print("\n" + "=" * 60)
    print("PERFORMING ERROR ANALYSIS")
    print("=" * 60)
    
    try:
        from evaluation.error_analysis import analyze_errors, plot_error_distribution
        
        df_errors, confusion_pairs = analyze_errors(
            model, test_loader, device, class_names, str(output_dirs["metrics"])
        )
        
        if len(df_errors) > 0:
            error_plot_path = output_dirs["figures"] / "error_distribution.png"
            plot_error_distribution(df_errors, error_plot_path, class_names)
            
            logger.info(f"Found {len(df_errors)} error samples ({len(df_errors)/len(test_dataset)*100:.2f}%)")
            logger.info("Confusion pairs:")
            for _, row in confusion_pairs.iterrows():
                logger.info(f"  {row['true_label']} → {row['pred_label']}: {row['count']} errors")
        else:
            logger.info("✓ No errors found! Model achieved perfect predictions on test set")
            
    except ImportError as e:
        logger.warning(f"Error analysis modules not available: {e}")
        print("Skipping error analysis...")
    except Exception as e:
        logger.error(f"Error in error analysis: {e}")
    
    # 3. GradCAM visualizations
    print("\n" + "=" * 60)
    print("GENERATING GRADCAM VISUALIZATIONS")
    print("=" * 60)

    try:
        from evaluation.xai_analysis import generate_gradcam_visualization

        generate_gradcam_visualization(
            model, test_loader, device, class_names, 
            str(output_dirs["figures"]), max_samples=10
        )

    except ImportError as e:
        logger.warning(f"GradCAM modules not available: {e}")
        print("Skipping GradCAM visualization...")
    except Exception as e:
        logger.error(f"Error in GradCAM visualization: {e}")

    # EDA moved to src/datasets/EDA/ and is NOT executed during training by default.
    # If you want to run dataset-level EDA, call datasets.EDA.eda_analysis.run_basic_eda separately.

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
    except Exception as e:
        logger.warning(f"Could not generate training summary: {e}")
    
    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Best model saved to: {save_path}")
    logger.info(f"✓ Training history saved to: {history_path}")
    logger.info(f"✓ Figures saved to: {output_dirs['figures']}")
    logger.info(f"✓ CSV summaries saved to: {output_dirs['csv']}")
    logger.info(f"✓ Logs saved to: {log_file}")
    logger.info("")
    logger.info(f"📊 FINAL TEST RESULTS:")
    logger.info(f"   Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"   Test F1 Score: {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    logger.info(f"   Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"   Test Recall: {test_metrics['recall']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()