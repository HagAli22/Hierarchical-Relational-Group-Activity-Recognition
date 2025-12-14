"""
Generic Distributed Data Parallel (DDP) Trainer
================================================
A unified training module for multi-GPU training across different models.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from torch import amp
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
        
        # Enable cuDNN benchmark for faster training with fixed input sizes
        torch.backends.cudnn.benchmark = True
        
        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(
            self.config.log_dir, 
            self.config.model_name.replace('.pth', ''),
            timestamp
        )
        logger = setup_logger(log_dir, self.config.model_name, rank)
        
        if rank == 0:
            logger.info(f"Starting {self.config.model_name.replace('.pth', '')} Training")
            logger.info(f"Initialized DDP with world_size={world_size}")
            logger.info(f"Using device: cuda:{rank}")
        
        # Create model
        model = self.model_fn()
        model = model.to(device)
        #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
            pin_memory=self.config.pin_memory, num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, sampler=val_sampler,
            collate_fn=self.collate_fn,
            pin_memory=self.config.pin_memory, num_workers=self.config.num_workers
        )
        
        if rank == 0:
            logger.info(f"Training dataset size: {len(self.train_dataset)}")
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
            logger.info(f"Batch size: {self.config.batch_size}, Num epochs: {self.config.num_epochs}")
            logger.info(f"Optimizer: {self.config.optimizer.upper()}, LR: {self.config.learning_rate}")
        
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
        
        if rank == 0:
            logger.info(f"Checkpoint directory: {checkpoint_dir}")
            logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            train_sampler.set_epoch(epoch)
            
            if rank == 0:
                logger.info(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
            
            # Get raw training metrics from this GPU
            train_total_loss, train_correct, train_total = self._train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch, logger, writer, rank
            )
            
            # Aggregate training metrics across all GPUs
            train_metrics = torch.tensor([train_total_loss, train_correct, train_total], dtype=torch.float32, device=device)
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
            
            # Calculate global training metrics
            train_loss = train_metrics[0].item() / train_metrics[2].item() if train_metrics[2].item() > 0 else 0
            train_acc = 100.0 * train_metrics[1].item() / train_metrics[2].item() if train_metrics[2].item() > 0 else 0
            
            if rank == 0:
                logger.info("Running validation...")
            
            # Get raw validation metrics from this GPU
            val_total_loss, val_correct, val_total = self.validate_fn(model, val_loader, criterion, device)
            
            # Aggregate metrics across all GPUs using all_reduce
            # Create tensors for aggregation
            metrics_tensor = torch.tensor([val_total_loss, val_correct, val_total], dtype=torch.float32, device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            
            # Calculate global metrics
            global_total_loss = metrics_tensor[0].item()
            global_correct = metrics_tensor[1].item()
            global_total = metrics_tensor[2].item()
            
            # Compute proper averages
            avg_val_loss = global_total_loss / global_total if global_total > 0 else 0
            val_acc = 100.0 * global_correct / global_total if global_total > 0 else 0
            
            # Scheduler step and check for LR change
            old_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step(avg_val_loss) if self.config.scheduler_type == 'reduce_on_plateau' else scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # Logging and checkpointing (rank 0 only)
            if rank == 0:
                logger.info(f"Validation completed - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
                
                # Log LR change
                if old_lr != new_lr:
                    logger.info(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(model, optimizer, epoch, avg_val_loss, val_acc, checkpoint_dir, scheduler, scaler)
                    logger.info(f"Checkpoint saved for epoch {epoch+1} at {checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
                
                # Epoch summary
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} Results:")
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                logger.info(f"  Learning Rate: {new_lr:.6f}")
                logger.info("-" * 60)
                
                # TensorBoard
                writer.add_scalars('Loss', {'train': train_loss, 'val': avg_val_loss}, epoch)
                writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
                writer.add_scalar('Learning_Rate', new_lr, epoch)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_best_model(model, checkpoint_dir)
                    logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
                    logger.info(f"Best model saved to: {checkpoint_dir}/{self.config.model_name}")
            
            if early_stopping and rank == 0 and early_stopping(avg_val_loss):
                logger.info("Early stopping triggered!")
                break
            
            dist.barrier()
        
        # Final evaluation
        if rank == 0:
            logger.info("Training completed!")
            logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")
            
            if self.config.generate_final_report:
                logger.info("Loading best model for final evaluation...")
                best_model_path = os.path.join(checkpoint_dir, self.config.model_name)
                if os.path.exists(best_model_path):
                    model.module.load_state_dict(torch.load(best_model_path, map_location=device))
                
                generate_evaluation_report(
                    model, val_loader, criterion, device, log_dir,
                    self.config.model_name, self.config.num_classes,
                    self.config.class_names, logger
                )
            
            writer.close()
            logger.info("=" * 60)
            logger.info(f"All results saved to: {log_dir}")
            logger.info("Training completed successfully")
        
        dist.destroy_process_group()
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, scaler, device, epoch, logger, writer, rank):
        """Train for one epoch. Returns raw values for DDP aggregation."""
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            if self.config.use_amp and scaler:
                with amp.autocast('cuda', dtype=torch.float16):
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
            
            batch_size = data.size(0)
            running_loss += loss.item() * batch_size
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += batch_size
            
            # Log batch progress with current accuracy (local to this GPU)
            if rank == 0 and (batch_idx + 1) % self.config.log_every_n_batches == 0:
                current_acc = 100.0 * correct / total if total > 0 else 0
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
        
        # Return raw values for DDP aggregation
        return running_loss, correct, total
    
    def _default_validate(self, model, val_loader, criterion, device):
        """
        Default validation function.
        Returns raw values (total_loss, correct, total) for proper DDP aggregation.
        """
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device, dtype=torch.long)
                output = model(data)
                batch_size = target.size(0)
                # Accumulate loss * batch_size for proper averaging later
                total_loss += criterion(output, target).item() * batch_size
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += batch_size
        
        # Return raw values for DDP aggregation
        return total_loss, correct, total
    
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