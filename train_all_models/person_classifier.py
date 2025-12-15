"""
Person Classifier Training Script
===============================================
Stage 1 of the HRN pipeline: Individual player action recognition.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import albumentations as albu
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from configs.config_loader import load_config
from data.data_loader import Person_classifier_loaders
from models import Person_Classifer
from train_utils.ddp_trainer import DDPTrainer, TrainingConfig


CONFIG_PATH = "configs/person_config.yaml"

# Kaggle paths
IS_KAGGLE = '/kaggle' in os.getcwd() or os.path.exists('/kaggle')
KAGGLE_OUTPUT = "/kaggle/working" if IS_KAGGLE else "."

# Global variable for num_classes (needed for pickling)
NUM_CLASSES = 9


def create_model():
    """Model factory function - must be defined at module level for multiprocessing."""
    return Person_Classifer(num_classes=NUM_CLASSES)


def get_transforms():
    """Get train and validation transforms."""
    train_transform = albu.Compose([
        albu.Resize(224, 224),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
        albu.OneOf([
            albu.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        ], p=0.5),
        albu.OneOf([
            albu.GaussianBlur(blur_limit=(3, 5)),
            albu.GaussNoise(var_limit=(10, 30)),
        ], p=0.3),
        albu.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
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
    
    print("Loading Person Classifier Configuration...")
    config = load_config(CONFIG_PATH)
    
    # Update global NUM_CLASSES from config
    NUM_CLASSES = config.model.person_activity.num_classes
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = Person_classifier_loaders(
        videos_path="/kaggle/input/volleyball/volleyball_/videos",
        annot_path="data/annot_all.pkl",
        split=config.data.train_split,
        transform=train_transform
    )
    
    val_dataset = Person_classifier_loaders(
        videos_path="/kaggle/input/volleyball/volleyball_/videos",
        annot_path="data/annot_all.pkl",
        split=config.data.val_split,
        transform=val_transform
    )
    
    test_dataset = Person_classifier_loaders(
        videos_path="/kaggle/input/volleyball/volleyball_/videos",
        annot_path="data/annot_all.pkl",
        split=config.data.test_split,
        transform=val_transform
    )
    
    # Create training config
    training_config = TrainingConfig(
        model_name=config.training.person_activity.model_name,
        num_classes=NUM_CLASSES,
        num_epochs=config.training.person_activity.num_epochs,
        batch_size=config.model.person_activity.batch_size,
        learning_rate=config.training.person_activity.learning_rate,
        weight_decay=config.training.person_activity.weight_decay,
        label_smoothing=getattr(config.training.person_activity, 'label_smoothing', 0.0),
        optimizer=config.training.person_activity.optimizer,
        use_scheduler=getattr(config.training, 'use_scheduler', True),
        scheduler_type=getattr(config.training, 'scheduler_type', 'reduce_on_plateau'),
        scheduler_patience=getattr(config.training, 'scheduler_patience', 5),
        scheduler_factor=getattr(config.training, 'scheduler_factor', 0.1),
        use_amp=getattr(config.training.person_activity, 'use_amp', True),
        checkpoint_dir=f"{KAGGLE_OUTPUT}/checkpoints/person_classifier" if IS_KAGGLE else config.training.person_activity.checkpoint_dir,
        log_dir=f"{KAGGLE_OUTPUT}/results/person_classifier" if IS_KAGGLE else "reslutes_and_logs/person_classifier",
        num_workers=getattr(config.model.person_activity, 'num_workers', 4),
        pin_memory=getattr(config.model.person_activity, 'pin_memory', True),
        resume_from_checkpoint=getattr(config.training.person_activity, 'resume_from_checkpoint', False),
        checkpoint_path=getattr(config.training.person_activity, 'checkpoint_path', ''),
        generate_final_report=True,
        class_names=getattr(config.model.person_activity, 'class_names', None),
    )
    
    # Create trainer and run
    trainer = DDPTrainer(
        model_fn=create_model,  # Use module-level function
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        collate_fn=None,
        test_dataset=test_dataset
    )
    
    trainer.run()


if __name__ == '__main__':
    main()
