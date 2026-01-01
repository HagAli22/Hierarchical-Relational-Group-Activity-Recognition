"""
Custom DDP Trainer for GAT Models with Entropy Regularization
==============================================================
Extends the base DDPTrainer to handle attention entropy loss.
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
from datetime import datetime

from train_utils.train_utils import (
    EarlyStopping,
    setup_logger,
    create_optimizer,
    create_scheduler,
    generate_evaluation_report,
)
from train_utils.ddp_trainer import TrainingConfig


class GATTrainer:
    """DDP Trainer for GAT models with attention entropy regularization."""
    
    def __init__(
        self,
        model_fn,
        train_dataset,
        val_dataset,
        config: TrainingConfig,
        collate_fn=None,
        test_dataset=None,
    ):
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.collate_fn = collate_fn
        
    def run(self):
        """Start DDP training across all available GPUs."""
        world_size = torch.cuda.device_count()
        
        if world_size < 1:
            raise RuntimeError("No CUDA devices available for DDP training")
        
        print(f"Starting GAT DDP training with {world_size} GPU(s)")
        
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        mp.spawn(self._train_worker, args=(world_size,), nprocs=world_size, join=True)
    
    def _train_worker(self, rank: int, world_size: int):
        """Training worker for each GPU process."""
        dist.init_process_group(
            backend=self.config.backend,
            init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
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
            logger.info(f"Starting GAT Training with Entropy Regularization")
            logger.info(f"Initialized DDP with world_size={world_size}")
        
        # Create model
        model = self.model_fn()
        model = model.to(device)
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
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, sampler=val_sampler,
            collate_fn=self.collate_fn
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
        
        best_val_loss = float('inf')
        
        if rank == 0:
            logger.info(f"Checkpoint directory: {checkpoint_dir}")
            logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            train_sampler.set_epoch(epoch)
            
            if rank == 0:
                logger.info(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
            
            # Train epoch with entropy loss
            train_loss, train_ce_loss, train_entropy_loss, train_correct, train_total = self._train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch, logger, rank
            )
            
            # Aggregate training metrics
            train_metrics = torch.tensor([train_loss, train_ce_loss, train_entropy_loss, train_correct, train_total], 
                                        dtype=torch.float32, device=device)
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
            
            avg_train_loss = train_metrics[0].item() / train_metrics[4].item()
            avg_train_ce = train_metrics[1].item() / train_metrics[4].item()
            avg_train_entropy = train_metrics[2].item() / train_metrics[4].item()
            train_acc = 100.0 * train_metrics[3].item() / train_metrics[4].item()
            
            if rank == 0:
                logger.info("Running validation...")
            
            # Validate
            val_loss, val_ce_loss, val_entropy_loss, val_correct, val_total = self._validate(
                model, val_loader, criterion, device
            )
            
            # Aggregate validation metrics
            val_metrics = torch.tensor([val_loss, val_ce_loss, val_entropy_loss, val_correct, val_total], 
                                       dtype=torch.float32, device=device)
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
            
            avg_val_loss = val_metrics[0].item() / val_metrics[4].item()
            avg_val_ce = val_metrics[1].item() / val_metrics[4].item()
            avg_val_entropy = val_metrics[2].item() / val_metrics[4].item()
            val_acc = 100.0 * val_metrics[3].item() / val_metrics[4].item()
            
            # Scheduler step
            old_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step(avg_val_loss) if self.config.scheduler_type == 'reduce_on_plateau' else scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # Logging (rank 0 only)
            if rank == 0:
                logger.info(f"Validation completed - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
                
                if old_lr != new_lr:
                    logger.info(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} Results:")
                logger.info(f"  Train - Total: {avg_train_loss:.4f}, CE: {avg_train_ce:.4f}, Entropy: {avg_train_entropy:.6f}, Acc: {train_acc:.2f}%")
                logger.info(f"  Val   - Total: {avg_val_loss:.4f}, CE: {avg_val_ce:.4f}, Entropy: {avg_val_entropy:.6f}, Acc: {val_acc:.2f}%")
                logger.info(f"  Learning Rate: {new_lr:.6f}")
                logger.info("-" * 60)
                
                # TensorBoard
                writer.add_scalars('Loss/Total', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
                writer.add_scalars('Loss/CE', {'train': avg_train_ce, 'val': avg_val_ce}, epoch)
                writer.add_scalars('Loss/Entropy', {'train': avg_train_entropy, 'val': avg_val_entropy}, epoch)
                writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
                writer.add_scalar('Learning_Rate', new_lr, epoch)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_best_model(model, checkpoint_dir)
                    self._save_checkpoint(model, optimizer, epoch, avg_val_loss, val_acc, checkpoint_dir, scheduler, scaler)
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
            
            if self.config.generate_final_report and self.test_dataset is not None:
                logger.info("Loading best model for final evaluation...")
                best_model_path = os.path.join(checkpoint_dir, self.config.model_name)
                if os.path.exists(best_model_path):
                    model.module.load_state_dict(torch.load(best_model_path, map_location=device))
                
                logger.info("Running final evaluation on TEST set...")
                test_loader = DataLoader(
                    self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                    collate_fn=self.collate_fn,
                    pin_memory=self.config.pin_memory, num_workers=self.config.num_workers
                )
                
                # Custom evaluation for GAT
                self._final_evaluation(model, test_loader, criterion, device, log_dir, logger)
            
            writer.close()
            logger.info("=" * 60)
            logger.info(f"All results saved to: {log_dir}")
        
        dist.destroy_process_group()
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, scaler, device, epoch, logger, rank):
        """Train for one epoch with entropy regularization."""
        model.train()
        total_loss, total_ce_loss, total_entropy_loss = 0.0, 0.0, 0.0
        correct, total = 0, 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            if self.config.use_amp and scaler:
                with amp.autocast('cuda', dtype=torch.float16):
                    output, entropy_loss = model(data)
                    ce_loss = criterion(output, target)
                    loss = ce_loss + entropy_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output, entropy_loss = model(data)
                ce_loss = criterion(output, target)
                loss = ce_loss + entropy_loss
                loss.backward()
                optimizer.step()
            
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_ce_loss += ce_loss.item() * batch_size
            total_entropy_loss += entropy_loss.item() * batch_size
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += batch_size
            
            if rank == 0 and (batch_idx + 1) % self.config.log_every_n_batches == 0:
                current_acc = 100.0 * correct / total
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
        
        return total_loss, total_ce_loss, total_entropy_loss, correct, total
    
    def _validate(self, model, val_loader, criterion, device):
        """Validate with entropy loss tracking."""
        model.eval()
        total_loss, total_ce_loss, total_entropy_loss = 0.0, 0.0, 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device, dtype=torch.long)
                output, entropy_loss = model(data)
                ce_loss = criterion(output, target)
                loss = ce_loss + entropy_loss
                
                batch_size = target.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_entropy_loss += entropy_loss.item() * batch_size
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += batch_size
        
        return total_loss, total_ce_loss, total_entropy_loss, correct, total
    
    def _final_evaluation(self, model, test_loader, criterion, device, log_dir, logger):
        """Final evaluation on test set."""
        model.eval()
        all_preds, all_targets = [], []
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device, dtype=torch.long)
                output, _ = model(data)
                loss = criterion(output, target)
                
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                total_loss += loss.item() * target.size(0)
                correct += preds.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        logger.info(f"Final Test Loss: {avg_loss:.4f}")
        logger.info(f"Final Test Accuracy: {accuracy:.2f}%")
        
        # Save confusion matrix and classification report
        import numpy as np
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(all_targets, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {log_dir}/confusion_matrix.png")
        
        report = classification_report(all_targets, all_preds, 
                                       target_names=self.config.class_names if self.config.class_names else None)
        with open(os.path.join(log_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        logger.info(f"Classification report saved to {log_dir}/classification_report.txt")
    
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
