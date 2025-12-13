"""
Training Utilities
==================
Common utilities for training models including logging, checkpointing,
early stopping, and evaluation reporting.
"""

import os
import torch
import logging
import numpy as np
from datetime import datetime
from typing import Optional, List

# Optional imports for evaluation reports
try:
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric: float) -> bool:
        score = -val_metric if self.mode == 'min' else val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def setup_logger(log_dir: str, name: str = "train", rank: int = 0) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        log_dir: Directory to save log files
        name: Logger name
        rank: Process rank (only rank 0 logs to file in DDP)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"{name}_rank{rank}" if rank > 0 else name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler for all ranks
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler only for rank 0
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(os.path.join(log_dir, f'training_{timestamp}.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_path)


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Handle DataParallel wrapping
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def create_optimizer(model, optimizer_name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    """Create optimizer based on name."""
    opt_name = optimizer_name.lower()
    
    if opt_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # SGD
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)


def create_scheduler(optimizer, scheduler_type: str, num_epochs: int, 
                     patience: int = 5, factor: float = 0.1, min_lr: float = 1e-6):
    """Create learning rate scheduler based on type."""
    if scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )
    else:  # step
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=patience, gamma=factor
        )


def save_training_summary(log_dir: str, model_name: str, config_dict: dict, 
                          final_loss: float, final_accuracy: float):
    """Save training summary to text file."""
    summary_path = os.path.join(log_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"TRAINING SUMMARY - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model Configuration:\n")
        for key, value in config_dict.items():
            f.write(f"  - {key}: {value}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"  - Validation Loss: {final_loss:.4f}\n")
        f.write(f"  - Validation Accuracy: {final_accuracy:.2f}%\n")
    return summary_path


def save_confusion_matrix(targets: np.ndarray, preds: np.ndarray, log_dir: str,
                          model_name: str, class_names: Optional[List[str]] = None,
                          num_classes: int = 8):
    """Save confusion matrix as PNG image."""
    if not SKLEARN_AVAILABLE:
        return None
    
    cm = confusion_matrix(targets, preds)
    
    plt.figure(figsize=(10, 8))
    
    labels = class_names if class_names else [str(i) for i in range(num_classes)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name.replace(".pth", "")}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(log_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    return cm_path


def save_classification_report_file(targets: np.ndarray, preds: np.ndarray, log_dir: str,
                                    model_name: str, accuracy: float,
                                    class_names: Optional[List[str]] = None):
    """Save classification report as text file."""
    if not SKLEARN_AVAILABLE:
        return None
    
    report = classification_report(targets, preds, target_names=class_names, digits=4)
    
    report_path = os.path.join(log_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"CLASSIFICATION REPORT - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Final Validation Accuracy: {accuracy:.2f}%\n\n")
        f.write(f"Per-Class Metrics:\n")
        f.write(f"{'-'*60}\n")
        f.write(report)
    return report_path


def generate_evaluation_report(model, val_loader, criterion, device, log_dir: str,
                               model_name: str, num_classes: int,
                               class_names: Optional[List[str]] = None,
                               logger: Optional[logging.Logger] = None):
    """
    Generate complete evaluation report including confusion matrix and classification report.
    
    Returns:
        dict with final_loss, final_accuracy, and paths to saved files
    """
    model.eval()
    all_preds = []
    all_targets = []
    val_loss = 0.0
    correct = 0
    total = 0
    
    if logger:
        logger.info("Running final evaluation...")
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device, dtype=torch.long)
            output = model(data)
            
            loss = criterion(output, target)
            val_loss += loss.item()
            
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    final_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    final_accuracy = 100.0 * correct / total if total > 0 else 0
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if logger:
        logger.info(f"Final Validation Loss: {final_loss:.4f}")
        logger.info(f"Final Validation Accuracy: {final_accuracy:.2f}%")
    
    # Save reports
    config_dict = {
        'Model Name': model_name,
        'Num Classes': num_classes,
    }
    summary_path = save_training_summary(log_dir, model_name, config_dict, final_loss, final_accuracy)
    
    cm_path = None
    report_path = None
    
    if SKLEARN_AVAILABLE:
        cm_path = save_confusion_matrix(all_targets, all_preds, log_dir, model_name, class_names, num_classes)
        report_path = save_classification_report_file(all_targets, all_preds, log_dir, model_name, final_accuracy, class_names)
        
        if logger:
            logger.info(f"Confusion matrix saved to {cm_path}")
            logger.info(f"Classification report saved to {report_path}")
    
    return {
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'summary_path': summary_path,
        'confusion_matrix_path': cm_path,
        'classification_report_path': report_path
    }
