# train_ddp.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from torch import amp
import logging
import numpy as np

import os
import sys
from PIL import __version__ as PILLOW_VERSION
import torch
import torch.nn as nn
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch import amp
from torch.utils.data import DataLoader , Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torchvision import models
import pickle
from PIL import Image
import numpy as np
from collections import Counter
import logging
from datetime import datetime
import yaml
from pathlib import Path

root_dataset = '/kaggle/input/volleyball/videos_sample'
CONFIG_PATH="/kaggle/input/vggcon/b3_config.yaml"
# ---- افترض أن الدوال/الكلاسات دي موجودة في سكربتك ----
# from your_module import (load_yaml_config, CONFIG_PATH, setup_logger,
#                          Person_Classifer, get_B3_A_loaders,
#                          validate_model, EarlyStopping)
# ------------------------------------------------------

def setup_basic_logger():
    logger = logging.getLogger("train_ddp")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger

class Config:
    def __init__(self, config_dict):
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})
        self.data = config_dict.get("data", {})
        self.evaluation=config_dict.get("evaluation",{})
        self.experiment = config_dict.get("experiment", {})

def load_yaml_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config(config_dict)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
class BoxInfo:
    def __init__(self, line):
        #line like : 0 361 469 413 569 24735 0 1 0 setting
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated
    
    sys.modules['boxinfo'] = sys.modules[__name__]

class Person_classifier_loaders(Dataset):
    def __init__(self, videos_path, annot_path, split, transform=None, logger=None):
        self.samples = []
        self.transform = transform
        self.categories_dct = {
            'waiting': 0,
            'setting': 1,
            'digging': 2,
            'falling': 3,
            'spiking': 4,
            'blocking': 5,
            'jumping': 6,
            'moving': 7,
            'standing': 8
        }

        self.data = []
        
        
        try:
            with open(annot_path, 'rb') as file:
                videos_annot = pickle.load(file)
        except Exception as e:
            raise

        box_count = 0
        for idx in split:
            video_annot = videos_annot[str(idx)]

            for clip in video_annot.keys():
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())

                for frame_id, boxes in dir_frames:
                    #print(type(frame_id), type(clip))
                    # framestr= str(frame_id)
                    # if framestr == clip:
                    #     print("s",frame_id, clip)
                    image_path = f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    image_path = os.path.join(image_path)

                    for box_info in boxes:
                        x1, y1, x2, y2 = box_info.box
                        category = box_info.category
                        box_count += 1

                        self.data.append(
                            {
                                'frame_path': f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                                'box': (x1, y1, x2, y2),
                                'category': category
                            }
                        )
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        num_classes=len(self.categories_dct)
        category=sample['category']
        label_idx=torch.zeros(num_classes)
        label_idx[self.categories_dct[category]] = 1
        label_idx = self.categories_dct[sample['category']]
        # num_classes = len(self.categories_dct)
        # labels = torch.zeros(num_classes)
        #label_idx = self.categories_dct[sample['category']]

        # labels[self.categories_dct[sample['category']]] = 1

        image = Image.open(sample['frame_path']).convert('RGB')
        x1, y1, x2, y2 = sample['box']
        cropped_image = image.crop((x1, y1, x2, y2))
    
        if self.transform:
            cropped_image = self.transform(image=np.array(cropped_image))['image']

        return cropped_image, label_idx


class Person_Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Person_Classifer, self).__init__()
        
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape      # x.shape => batch, channals , hight, width
        x = self.resnet50(x)      # (batch, 2048, 1 , 1)
        x = x.view(b, -1)         # (batch, 2048)
        x = self.fc(x)            # (batch, num_class)          
        return x



class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
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

def validate_model(model, val_loader, criterion, device):
    """Function to validate the model and return validation loss and accuracy"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device, dtype=torch.long)
            output = model(data)

            # Calculate validation loss
            loss = criterion(output, target)
            val_loss += loss.item()

            predicted = output.argmax(dim=1)
            # _, target_class = target.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # _, predicted = output.max(1)
            # _, target_class = target.max(1)
            # total += target.size(0)
            # correct += predicted.eq(target_class).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = (correct / total) * 100

    return avg_val_loss, accuracy


# Setup logging
def setup_logger():
    """Setup logger with file and console handlers"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/training_{timestamp}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
def train_worker(rank, world_size, CONFIG_PATH):
    """
    كل عملية (واحدة لكل GPU) تنفذ هذه الدالة.
    rank: رقم العملية (0 .. world_size-1)
    world_size: عدد العمليات = عدد GPUs
    """
    logger = setup_basic_logger()
    # === Distributed init ===
    dist.init_process_group(backend="nccl",
                            init_method="tcp://127.0.0.1:29500",
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        logger.info(f"Initialized DDP with world_size={world_size}")

    # === Load config & prepare ===
    config = load_yaml_config(CONFIG_PATH)
    train_split = config.data['train_split']
    val_split   = config.data['val_split']

    # Transforms (افترضت أنها كما في سكربتك)
    train_preprocess = ...  # ضع هنا التعريف أو استورده من سكربتك
    val_preprocess = ...    # نفس الشيء

    # === Model ===
    model = Person_Classifer(num_classes=config.model['person_activity']['num_classes'])
    model = model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # إن احتجت سنك batchnorm
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # === Optimizer ===
    opt_cfg = config.training['person_activity']
    if opt_cfg['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.get('learning_rate', 4e-4),
                                      weight_decay=opt_cfg.get('weight_decay', 1.0),
                                      betas=(0.9, 0.999))
    elif opt_cfg['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg.get('learning_rate', 1e-3),
                                     weight_decay=opt_cfg.get('weight_decay', 0.0))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg.get('learning_rate', 1e-3),
                                    weight_decay=opt_cfg.get('weight_decay', 0.0),
                                    momentum=opt_cfg.get('momentum', 0.9))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=3, factor=0.1, min_lr=1e-6)

    early_stopping = EarlyStopping(patience=7, min_delta=0.0001, mode='min')

    # === Dataset & DistributedSampler & DataLoader ===
    # Use same dataset creation you have; هنا استخدمت get_B3_A_loaders كما في سكربتك
    train_dataset = get_B3_A_loaders(
        videos_path=f"/kaggle/input/volleyball/volleyball_/videos",
        annot_path=f"/kaggle/input/b5-aanno/annot_all.pkl",
        split=train_split,
        transform=train_preprocess
    )

    val_dataset = get_B3_A_loaders(
        videos_path=f"/kaggle/input/volleyball/volleyball_/videos",
        annot_path=f"/kaggle/input/b5-aanno/annot_all.pkl",
        split=val_split,
        transform=val_preprocess
    )

    # Global batch from config (إذا غير متوفر استخدم 256)
    global_batch = config.model['person_activity'].get('batch_size', 256)
    # per_proc_batch = max(1, global_batch // world_size)  # batch لكل عملية
    per_proc_batch = global_batch
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=per_proc_batch, sampler=train_sampler,
                              #num_workers=config.model['person_activity'].get('num_workers', 4),
                              #pin_memory=config.model['person_activity'].get('pin_memory', True)
                             )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=per_proc_batch, sampler=val_sampler,
                            #num_workers=config.model['person_activity'].get('num_workers', 4),
                            #pin_memory=config.model['person_activity'].get('pin_memory', True)
                           )

    if rank == 0:
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        logger.info(f"Per-process batch size: {per_proc_batch}")

    # === Loss and AMP scaler ===
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.01).to(device)
    use_amp = config.training['person_activity'].get('use_amp', False)
    scaler = GradScaler() if use_amp else None

    # === Training loop (مختصر ومماثل لكودك) ===
    num_epochs = config.training['person_activity'].get('epochs', 1)
    best_val_loss = float('inf')
    model_name = config.training['person_activity'].get('model_name', 'best_model.pth')
    checkpoint_dir = config.training['person_activity'].get('checkpoint_dir', './checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # مهم لتجزئة البيانات عند التدريب التكراري
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(data)
                    loss = criterion(outputs, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

            if rank == 0 and (batch_idx + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_sampler)  # استخدم طول السامبلر (لكل عملية)
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # === Validation (بمعاملة model.eval) ===
        # استخدام validate_model لكن تأكد أنه لا يقوم بعمليات توزيع داخلية
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        # reduce val_loss across processes: نجمع القيم من كل عملية ونقسم
        val_loss_tensor = torch.tensor(val_loss).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        # average over processes
        avg_val_loss = (val_loss_tensor.item() / world_size)

        # Scheduler step (only use avg_val_loss on rank 0 or pass avg)
        scheduler.step(avg_val_loss)

        # Save checkpoint ONLY on rank 0
        if rank == 0:
            logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_acc
            }
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                           os.path.join(checkpoint_dir, model_name))
                logger.info(f"New best model saved with val_loss={best_val_loss:.4f}")

            # early stopping check (rank 0 handles)
            if early_stopping is not None:
                early_stopping(avg_val_loss)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered on rank 0. Breaking training loop.")
                    break

        # sync a barrier to make sure rank 0 saved before others proceed to next epoch
        dist.barrier()

    # النهاية: ديسكونكت
    dist.destroy_process_group()
    if rank == 0:
        logger.info("Training finished and process group destroyed.")

if __name__ == "__main__":
    # عدد الـ GPUs المتوفرة
    ngpus = torch.cuda.device_count()
    if ngpus <= 1:
        print("Less than 2 GPUs found — DDP still يعمل على جهاز واحد لكن الأداء منخفض.")
    # استبدل هذا بالمسار الحقيقي للـ config عندك
    CONFIG_PATH = "/kaggle/input/vggcon/b3_config.yaml"
    # ابدأ world_size عمليات (واحدة لكل GPU)
    mp.spawn(train_worker, args=(ngpus, CONFIG_PATH), nprocs=ngpus, join=True)