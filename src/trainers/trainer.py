import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import logging
from pathlib import Path

from trainers.metrics import calculate_metrics


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_path,
        logger=None,
        patience=4,
        grad_clip_norm=1.0,
        use_amp=True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = Path(save_path)
        self.logger = logger or logging.getLogger(__name__)
        self.best_f1 = 0.0
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.early_stop_counter = 0
        
        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        y_true = []
        y_pred = []

        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in loop:
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Xử lý metadata
            metadata = batch.get("metadata")
            if metadata is not None:
                if isinstance(metadata, dict):
                    # Nếu metadata vẫn là dict, bỏ qua
                    metadata = None
                else:
                    metadata = metadata.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images, input_ids, attention_mask, metadata_features=metadata)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images, input_ids, attention_mask, metadata_features=metadata)
                loss = self.criterion(outputs, labels)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            loop.set_postfix(loss=loss.item())

        metrics = calculate_metrics(y_true, y_pred)
        avg_loss = running_loss / len(self.train_loader)
        
        self.logger.info(f"Train Loss: {avg_loss:.4f}, Train Acc: {metrics['accuracy']:.4f}, Train F1: {metrics['f1']:.4f}")
        
        return avg_loss, metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        y_true = []
        y_pred = []

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            metadata = batch.get("metadata")
            if metadata is not None:
                if isinstance(metadata, dict):
                    metadata = None
                else:
                    metadata = metadata.to(self.device)

            outputs = self.model(images, input_ids, attention_mask, metadata_features=metadata)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        metrics = calculate_metrics(y_true, y_pred)
        avg_loss = running_loss / len(self.val_loader)
        
        self.logger.info(f"Val Loss: {avg_loss:.4f}, Val Acc: {metrics['accuracy']:.4f}, Val F1: {metrics['f1']:.4f}")
        
        return avg_loss, metrics

    def save_best_model(self, f1_score, epoch):
        if f1_score > self.best_f1:
            self.best_f1 = f1_score
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_f1': self.best_f1,
            }
            torch.save(checkpoint, self.save_path.parent / "full_checkpoint.pth")
            torch.save(self.model.state_dict(), self.save_path)
            
            self.logger.info(f"\n✓ Best model saved! (F1={f1_score:.4f})")

    def fit(self, epochs):
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Batch size: {self.train_loader.batch_size}")
        self.logger.info("=" * 60)
        
        for epoch in range(epochs):
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.logger.info(f"{'='*40}")
            
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_f1s.append(train_metrics['f1'])
            self.val_f1s.append(val_metrics['f1'])
            self.train_accs.append(train_metrics['accuracy'])
            self.val_accs.append(val_metrics['accuracy'])
            
            self.save_best_model(val_metrics["f1"], epoch + 1)
            
            if val_metrics["f1"] > self.best_f1:
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            self.logger.info(f"\n📊 Epoch {epoch + 1} Summary:")
            self.logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['f1']:.4f}")
            self.logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f} | Val F1:   {val_metrics['f1']:.4f}")
            self.logger.info(f"  Early stop counter: {self.early_stop_counter}/{self.patience}")
            self.logger.info(f"  Best Val F1: {self.best_f1:.4f}")
            
            if self.early_stop_counter >= self.patience:
                self.logger.info(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
                break
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Best Val F1: {self.best_f1:.4f}")
        self.logger.info("=" * 60)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_f1s': self.train_f1s,
            'val_f1s': self.val_f1s,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_f1': self.best_f1
        }