"""
B1-NoRelations Training Script
============================================
Baseline model for group activity recognition WITHOUT relational reasoning.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from configs.config_loader import load_config
from data.data_loader import GroupActivityDataset
from models import B1_NoRelations, collate_group_fn, Person_Classifer
from train_utils.ddp_trainer import DDPTrainer, TrainingConfig


CONFIG_PATH = "configs/non_temporal_model/B1NoRelations_config.yaml"

# Kaggle paths
IS_KAGGLE = '/kaggle' in os.getcwd() or os.path.exists('/kaggle')
KAGGLE_OUTPUT = "/kaggle/working" if IS_KAGGLE else "."

# Global variables for model creation (needed for pickling)
NUM_CLASSES = 8
PERSON_CLASSIFIER_PATH = "/kaggle/input/person-classifer/results/person_classifier/person_classifier_best/20251214_140908/checkpoints/person_classifier_best.pth"


def create_model():
    """Model factory function - must be defined at module level for multiprocessing."""

    person_model = Person_Classifer(num_classes=9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    person_checkpoint = torch.load(PERSON_CLASSIFIER_PATH, map_location=device, weights_only=True)
    
    # Handle both checkpoint formats (dict with 'model_state_dict' or direct state_dict)
    if isinstance(person_checkpoint, dict) and 'model_state_dict' in person_checkpoint:
        person_model.load_state_dict(person_checkpoint['model_state_dict'])
    else:
        person_model.load_state_dict(person_checkpoint)

    return B1_NoRelations(person_model, num_classes=NUM_CLASSES)


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
    global NUM_CLASSES, PERSON_CLASSIFIER_PATH
    
    print("Loading B1NoRelations Configuration...")
    config = load_config(CONFIG_PATH)
    
    # Update global variables from config
    NUM_CLASSES = config.model.group_activity.num_classes
    PERSON_CLASSIFIER_PATH = config.training.group_activity.person_classifier_path
    
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
        model_name=getattr(config.training.group_activity, 'model_name', 'B1NoRelations.pth'),
        num_classes=NUM_CLASSES,
        num_epochs=config.training.group_activity.num_epochs,
        batch_size=config.model.group_activity.batch_size,
        learning_rate=config.training.person_activity.learning_rate,
        weight_decay=config.training.person_activity.weight_decay,
        label_smoothing=getattr(config.training.group_activity, 'label_smoothing', 0.0),
        optimizer=config.training.person_activity.optimizer,
        use_scheduler=getattr(config.training, 'use_scheduler', True),
        scheduler_type=getattr(config.training, 'scheduler_type', 'reduce_on_plateau'),
        scheduler_patience=getattr(config.training, 'scheduler_patience', 5),
        scheduler_factor=getattr(config.training, 'scheduler_factor', 0.1),
        use_amp=getattr(config.training.group_activity, 'use_amp', True),
        checkpoint_dir=f"{KAGGLE_OUTPUT}/checkpoints/B1NoRelations" if IS_KAGGLE else config.training.person_activity.checkpoint_dir,
        log_dir=f"{KAGGLE_OUTPUT}/results/B1NoRelations" if IS_KAGGLE else "reslutes_and_logs/non_temporal_model",
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
