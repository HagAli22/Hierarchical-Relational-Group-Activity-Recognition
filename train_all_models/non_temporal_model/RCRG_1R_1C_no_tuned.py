"""
RCRG-1R-1C-!tuned Training Script
============================================
Same as RCRG-1R-1C model with 1 Relational layer and 1 Clique for group activity recognition but ImageNet-pretrained ResNet50 without fine-tuning.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from configs.config_loader import load_config
from data.data_loader import GroupActivityDataset
from models.non_temporal_model.RCRG_1R_1C_no_tuned import RCRG_1R_1C_no_tuned, collate_group_fn
from models import Person_Classifer
from train_utils.ddp_trainer import DDPTrainer, TrainingConfig


CONFIG_PATH = "configs/non_temporal_model/RCRG_1R_1C_no_tuned_config.yaml"

# Kaggle paths
IS_KAGGLE = '/kaggle' in os.getcwd() or os.path.exists('/kaggle')
KAGGLE_OUTPUT = "/kaggle/working" if IS_KAGGLE else "."

# Global variables for model creation (needed for pickling)
NUM_CLASSES = 8


def create_model():
    """Model factory function - must be defined at module level for multiprocessing."""

    return RCRG_1R_1C_no_tuned(num_classes=NUM_CLASSES)


def get_transforms():
    """Get train and validation transforms."""
    train_transform = albu.Compose([
        albu.Resize(224, 224),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.3),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = albu.Compose([
        albu.Resize(224, 224),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def main():
    global NUM_CLASSES
    
    print("Loading RCRG-1R-1C-!tuned Configuration...")
    config = load_config(CONFIG_PATH)
    
    # Update global variables from config
    NUM_CLASSES = config.model.group_activity.num_classes
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = GroupActivityDataset(
        videos_path="/kaggle/input/volleyball/volleyball_/videos",
        annot_path="data/annot_all.pkl",
        split=config.data.train_split,
        sec=False,
        transform=train_transform
    )
    
    val_dataset = GroupActivityDataset(
        videos_path="/kaggle/input/volleyball/volleyball_/videos",
        annot_path="data/annot_all.pkl",
        split=config.data.val_split,
        sec=False,
        transform=val_transform
    )
    
    test_dataset = GroupActivityDataset(
        videos_path="/kaggle/input/volleyball/volleyball_/videos",
        annot_path="data/annot_all.pkl",
        split=config.data.test_split,
        sec=False,
        transform=val_transform
    )
    
    # Create training config
    training_config = TrainingConfig(
        model_name=getattr(config.training.group_activity, 'model_name', 'RCRG_1R_1C_no_tuned.pth'),
        num_classes=NUM_CLASSES,
        num_epochs=config.training.group_activity.num_epochs,
        batch_size=config.model.group_activity.batch_size,
        learning_rate=config.training.group_activity.learning_rate,
        weight_decay=config.training.group_activity.weight_decay,
        label_smoothing=getattr(config.training.group_activity, 'label_smoothing', 0.0),
        optimizer=config.training.group_activity.optimizer,
        use_scheduler=getattr(config.training, 'use_scheduler', True),
        scheduler_type=getattr(config.training, 'scheduler_type', 'reduce_on_plateau'),
        scheduler_patience=getattr(config.training, 'scheduler_patience', 5),
        scheduler_factor=getattr(config.training, 'scheduler_factor', 0.1),
        use_amp=getattr(config.training.group_activity, 'use_amp', True),
        checkpoint_dir=f"{KAGGLE_OUTPUT}/checkpoints/RCRG_1R_1C_no_tuned" if IS_KAGGLE else config.training.group_activity.checkpoint_dir,
        log_dir=f"{KAGGLE_OUTPUT}/results/RCRG_1R_1C_no_tuned" if IS_KAGGLE else "reslutes_and_logs/non_temporal_model",
        log_every_n_batches=getattr(config.training.group_activity, 'log_every_n_batches', 20),
        num_workers=getattr(config.model.group_activity, 'num_workers', 4),
        pin_memory=getattr(config.model.group_activity, 'pin_memory', True),
        resume_from_checkpoint=getattr(config.training.group_activity, 'resume_from_checkpoint', False),
        checkpoint_path=getattr(config.training.group_activity, 'checkpoint_path', ''),
        generate_final_report=True,
        class_names=getattr(config.model.group_activity, 'class_names', None),
    )
    
    # Create trainer and run
    trainer = DDPTrainer(
        model_fn=create_model,  # Use module-level function
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        collate_fn=collate_group_fn,
        test_dataset=test_dataset
    )
    
    trainer.run()


if __name__ == '__main__':
    main()
