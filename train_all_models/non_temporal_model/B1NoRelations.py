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
from data.boxinfo import BoxInfo
from models import B1_NoRelations, collate_group_fn, Person_Classifer
from train_utils.ddp_trainer import DDPTrainer, TrainingConfig


CONFIG_PATH = "configs/non_temporal_model/B1NoRelations_config.yaml"

# Kaggle paths
IS_KAGGLE = '/kaggle' in os.getcwd() or os.path.exists('/kaggle')
KAGGLE_OUTPUT = "/kaggle/working" if IS_KAGGLE else "."


def get_transforms():
    """Get train and validation transforms."""
    train_transform = albu.Compose([
        albu.Resize(224, 224),
        albu.OneOf([
            albu.HorizontalFlip(p=1.0),
            albu.Rotate(limit=20, p=1.0),
            albu.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=20, p=1.0),
        ], p=0.6),
        albu.OneOf([
            albu.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.15, p=1.0),
            albu.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
            albu.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=1.0),
        ], p=0.7),
        albu.OneOf([
            albu.GaussianBlur(blur_limit=(3, 9), p=1.0),
            albu.MotionBlur(blur_limit=9, p=1.0),
            albu.GaussNoise(var_limit=(10, 60), p=1.0),
        ], p=0.6),
        albu.CoarseDropout(
            max_holes=12, max_height=20, max_width=20,
            min_holes=3, min_height=10, min_width=10, 
            p=0.4
        ),
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
    print("Loading B1NoRelations Configuration...")
    config = load_config(CONFIG_PATH)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets - use dot notation
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
    
    # Load pretrained person classifier path - use dot notation
    person_classifier_path = config.training.group_activity.person_classifier_path
    
    # Create training config - use dot notation
    training_config = TrainingConfig(
        model_name=getattr(config.training.group_activity, 'model_name', 'B1NoRelations.pth'),
        num_classes=config.model.group_activity.num_classes,
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
        use_amp=getattr(config.training.person_activity, 'use_amp', True),
        checkpoint_dir=f"{KAGGLE_OUTPUT}/checkpoints/B1NoRelations" if IS_KAGGLE else config.training.person_activity.checkpoint_dir,
        log_dir=f"{KAGGLE_OUTPUT}/results/B1NoRelations" if IS_KAGGLE else "reslutes_and_logs/non_temporal_model",
        num_workers=getattr(config.model.group_activity, 'num_workers', 4),
        pin_memory=getattr(config.model.group_activity, 'pin_memory', True),
        # Resume settings
        resume_from_checkpoint=getattr(config.training.group_activity, 'resume_from_checkpoint', False),
        checkpoint_path=getattr(config.training.group_activity, 'checkpoint_path', ''),
        # Final report settings
        generate_final_report=True,
        class_names=getattr(config.model.group_activity, 'class_names', None),
    )
    
    # Model factory function
    def create_model():
        # Load pretrained person classifier
        person_model = Person_Classifer(num_classes=9)
        if os.path.exists(person_classifier_path):
            person_model.load_state_dict(torch.load(person_classifier_path, map_location='cpu'))
            print(f"Loaded pretrained person classifier from {person_classifier_path}")
        else:
            print(f"Warning: Person classifier not found at {person_classifier_path}")
        
        # Create B1NoRelations model
        model = B1_NoRelations(num_classes=training_config.num_classes)
        return model
    
    # Create trainer and run
    trainer = DDPTrainer(
        model_fn=create_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        collate_fn=collate_group_fn
    )
    
    trainer.run()


if __name__ == '__main__':
    main()
