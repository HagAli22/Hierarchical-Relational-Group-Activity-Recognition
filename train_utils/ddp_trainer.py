"""
Generic Distributed Data Parallel (DDP) Trainer
================================================
A unified training module for multi-GPU training across different models.

Usage:
    from train_utils.ddp_trainer import DDPTrainer, TrainingConfig
    
    trainer = DDPTrainer(
        model_fn=lambda: MyModel(num_classes=8),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        collate_fn=my_collate_fn  # optional
    )
    trainer.run()
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Callable, Optional, List
from datetime import datetime

from train_utils.train_utils import (
    EarlyStopping,
    setup_logger,
    create_optimizer,
    create_scheduler,
    generate_evaluation_report,
)


@dataclass
class TrainingConfig:
    """Configuration for DDP training."""
    # Model settings
    model_name: str = "model"
    num_classes: int = 8
    
    # Training hyperparameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    
    # Optimizer settings
    optimizer: str = "adam"  # adam, adamw, sgd
    momentum: float = 0.9  # for SGD
    
    # Scheduler settings
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.1
    scheduler_min_lr: float = 1e-6
    
    # AMP settings
    use_amp: bool = True
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 2
    
    # Resume from checkpoint
    resume_from_checkpoint: bool = False
    checkpoint_path: str = ""
    
    # Final evaluation settings
    generate_final_report: bool = True
    class_names: Optional[List[str]] = None
    
    # Logging
    log_dir: str = "reslutes_and_logs"
    log_every_n_batches: int = 50
    
    # DDP settings
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    backend: str = "nccl"
    
    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True


class DDPTrainer:
    """Generic DDP Trainer for multi-GPU training."""
    
    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        train_dataset,
        val_dataset,
        config: TrainingConfig,
        collate_fn: Optional[Callable] = None,
        validate_fn: Optional[Callable] = None,
    ):
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.collate_fn = collate_fn
        self.validate_fn = validate_fn or self._default_validate
        
    def run(self):
        """Start DDP training across all available GPUs."""
        world_size = torch.cuda.device_count()
        
        if world_size < 1:
            raise RuntimeError("No CUDA devices available for DDP training")
        
        print(f"Starting DDP training with {world_size} GPU(s)")
        
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        mp.spawn(self._train_worker, args=(world_size,), nprocs=world_size, join=True)
    
    def _train_worker(self, rank: int, world_size: int):
        """Training worker for each GPU process."""
        # Initialize DDP
        dist.init_process_group(
            backend=self.config.backend,
            init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(
            self.config.log_dir, 
            self.config.model_name.replace('.pth', ''),
            timestamp
        )
        logger = setup_logger(log_dir, self.config.model_name, rank)
        
        if rank == 0:
            logger.info(f"Initialized DDP with world_size={world_size}")
        
        # Create model
        model = self.model_fn()
        model = model.to(device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(
            model, self.config.optimizer, self.config.learning_rate,
            self.config.weight_decay, self.config.momentum
        )
        scheduler = create_scheduler(
            optimizer, self.config.scheduler_type, self.config.num_epochs,
            self.config.scheduler_patience, self.config.scheduler_factor, self.config.scheduler_min_lr
        ) if self.config.use_scheduler else None
        
        # Create data loaders
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, sampler=train_sampler,
            collate_fn=self.collate_fn, 
            #num_workers=self.config.num_workers, pin_memory=self.config.pin_memory
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, sampler=val_sampler,
            collate_fn=self.collate_fn,
            #num_workers=self.config.num_workers, pin_memory=self.config.pin_memory
        )
        
        if rank == 0:
            logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Batch: {self.config.batch_size}")
        
        # Loss, AMP, Early stopping
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing).to(device)
        scaler = GradScaler() if self.config.use_amp else None
        early_stopping = EarlyStopping(
            self.config.early_stopping_patience, self.config.early_stopping_min_delta
        ) if self.config.use_early_stopping else None
        
        # TensorBoard and checkpoints
        writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Resume if needed
        start_epoch, best_val_loss = self._load_checkpoint(model, optimizer, scheduler, scaler, device, logger, rank)
        
        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            train_sampler.set_epoch(epoch)
            
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger, writer, rank)
            val_loss, val_acc = self.validate_fn(model, val_loader, criterion, device)
            
            # Reduce val_loss across processes
            val_loss_tensor = torch.tensor(val_loss).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / world_size
            
            # Scheduler step
            if scheduler:
                scheduler.step(avg_val_loss) if self.config.scheduler_type == 'reduce_on_plateau' else scheduler.step()
            
            # Logging and checkpointing (rank 0 only)
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | Train: {train_loss:.4f}/{train_acc:.2f}% | Val: {avg_val_loss:.4f}/{val_acc:.2f}% | LR: {current_lr:.6f}")
                
                writer.add_scalars('Loss', {'train': train_loss, 'val': avg_val_loss}, epoch)
                writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
                
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(model, optimizer, epoch, avg_val_loss, val_acc, checkpoint_dir, scheduler, scaler)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_best_model(model, checkpoint_dir)
                    logger.info(f"  New best model! Val Loss: {best_val_loss:.4f}")
            
            if early_stopping and rank == 0 and early_stopping(avg_val_loss):
                logger.info("Early stopping triggered!")
                break
            
            dist.barrier()
        
        # Final evaluation
        if rank == 0 and self.config.generate_final_report:
            logger.info("=" * 60)
            best_model_path = os.path.join(checkpoint_dir, self.config.model_name)
            if os.path.exists(best_model_path):
                model.module.load_state_dict(torch.load(best_model_path, map_location=device))
            
            generate_evaluation_report(
                model, val_loader, criterion, device, log_dir,
                self.config.model_name, self.config.num_classes,
                self.config.class_names, logger
            )
            writer.close()
            logger.info(f"Training completed! Results saved to: {log_dir}")
        
        dist.destroy_process_group()
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, scaler, device, epoch, logger, writer, rank):
        """Train for one epoch."""
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, dtype=torch.long, non_blocking=True)
            
            optimizer.zero_grad()
            
            if self.config.use_amp and scaler:
                with autocast(dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
            
            if rank == 0 and (batch_idx + 1) % self.config.log_every_n_batches == 0:
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        return running_loss / total if total > 0 else 0, 100.0 * correct / total if total > 0 else 0
    
    def _default_validate(self, model, val_loader, criterion, device):
        """Default validation function."""
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device, dtype=torch.long)
                output = model(data)
                val_loss += criterion(output, target).item()
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        
        return val_loss / len(val_loader) if len(val_loader) > 0 else 0, 100.0 * correct / total if total > 0 else 0
    
    def _save_checkpoint(self, model, optimizer, epoch, val_loss, val_acc, checkpoint_dir, scheduler=None, scaler=None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    def _save_best_model(self, model, checkpoint_dir):
        """Save best model weights."""
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, self.config.model_name))
    
    def _load_checkpoint(self, model, optimizer, scheduler, scaler, device, logger, rank):
        """Load checkpoint to resume training."""
        if not self.config.resume_from_checkpoint or not self.config.checkpoint_path:
            return 0, float('inf')
        
        if not os.path.exists(self.config.checkpoint_path):
            if rank == 0:
                logger.warning(f"Checkpoint not found: {self.config.checkpoint_path}")
            return 0, float('inf')
        
        checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, val_loss: {best_val_loss:.4f}")
        
        return start_epoch, best_val_loss