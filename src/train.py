import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import yaml
import os
from typing import Dict, Optional, Tuple

from data import create_dataloaders
from model import StutterTTS, StutterTTSLoss, create_model
from utils import (
    set_seed,
    load_config,
    setup_logging,
    LearningRateScheduler,
    AverageMeter,
    save_figure,
    plot_spectrogram,
    plot_attention,
    create_experiment_directory,
    calculate_gradient_norm
)

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: Dict, experiment_dir: Optional[str] = None):
        self.config = config
        
        # Create experiment directory
        if experiment_dir is None:
            experiment_dir = create_experiment_directory(config["training"]["experiment_dir"])
        self.experiment_dir = experiment_dir
        
        # Setup logging
        setup_logging(self.experiment_dir["logs"])
        self.writer = SummaryWriter(self.experiment_dir["logs"])
        
        # Setup device
        self.device = torch.device(config["training"]["device"])
        
        # Create model
        self.model = create_model(config).to(self.device)
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Create loss function
        self.criterion = StutterTTSLoss(config)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            betas=(float(config["training"]["beta1"]), float(config["training"]["beta2"])),
            eps=float(config["training"]["epsilon"]),
            weight_decay=float(config["training"].get("weight_decay", 0))
        )
        
        # Create learning rate scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            config["model"]["d_model"],
            config["training"]["warmup_steps"]
        )
        
        # Create data loaders
        self.train_loader, self.valid_loader, self.test_loader = create_dataloaders(config)
        
        # Initialize training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
        # Gradient clipping
        self.grad_clip = config["training"]["grad_clip_thresh"]
        
        # Training config
        self.log_interval = config["training"]["log_interval"]
        self.eval_interval = config["training"]["eval_interval"]
        self.save_interval = config["training"]["save_interval"]
        self.max_epochs = config["training"]["epochs"]
    
    def save_checkpoint(self, name: str = "latest"):
        """Save model checkpoint"""
        checkpoint_path = Path(self.experiment_dir["checkpoints"]) / f"{name}.pt"
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self.best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = AverageMeter()
        
        with tqdm(total=len(self.train_loader), desc=f'Epoch {self.epoch+1}') as pbar:
            for batch in self.train_loader:
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                mel_output, mel_postnet, stop_tokens = self.model(
                    phoneme_ids=batch["phoneme_ids"],
                    mel_target=batch["mel_spectrogram"],
                    reference_mel=batch.get("reference_mel")
                )
                
                # Calculate loss
                loss, loss_dict = self.criterion(
                    mel_output, mel_postnet, stop_tokens,
                    batch["mel_spectrogram"], batch["mel_masks"]
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                epoch_loss.update(loss.item())
                self.step += 1
                
                # Logging
                if self.step % self.log_interval == 0:
                    self._log_training_step(loss_dict, grad_norm, batch)
                
                # Validation
                if self.step % self.eval_interval == 0:
                    val_loss = self.validate()
                    self.model.train()
                
                # Save checkpoint
                if self.step % self.save_interval == 0:
                    self.save_checkpoint()
                
                # Update progress bar
                pbar.set_postfix(loss=f'{epoch_loss.avg:.4f}',
                               lr=f'{self.scheduler._get_lr():.6f}')
                pbar.update(1)
        
        return epoch_loss.avg
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation"""
        self.model.eval()
        val_loss = AverageMeter()
        
        for batch in tqdm(self.valid_loader, desc='Validation'):
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            mel_output, mel_postnet, stop_tokens = self.model(
                phoneme_ids=batch["phoneme_ids"],
                mel_target=batch["mel_spectrogram"],
                reference_mel=batch.get("reference_mel")
            )
            
            # Calculate loss
            loss, loss_dict = self.criterion(
                mel_output, mel_postnet, stop_tokens,
                batch["mel_spectrogram"], batch["mel_masks"]
            )
            
            val_loss.update(loss.item())
        
        # Log validation results
        self._log_validation(val_loss.avg, batch)
        
        return val_loss.avg
    
    def _log_training_step(self, loss_dict: Dict, grad_norm: float, batch: Dict):
        """Log training step information"""
        # Log losses
        for name, value in loss_dict.items():
            self.writer.add_scalar(f'train/{name}', value, self.step)
        
        # Log gradient norm
        self.writer.add_scalar('train/gradient_norm', grad_norm, self.step)
        
        # Log learning rate
        self.writer.add_scalar('train/learning_rate',
                             self.scheduler._get_lr(), self.step)
        
        # Log spectrograms
        if self.step % (self.log_interval * 10) == 0:
            idx = 0  # First item in batch
            fig = plot_spectrogram(batch["mel_spectrogram"][idx],
                                 title="Ground Truth Mel-spectrogram")
            self.writer.add_figure('train/mel_target', fig, self.step)
            
            # Get attention weights
            attentions = self.model.get_attention_weights()
            for name, weights in attentions.items():
                if weights:  # if not empty
                    fig = plot_attention(weights[-1][0],
                                      title=f"{name} (Last Layer)")
                    self.writer.add_figure(f'train/{name}', fig, self.step)
    
    def _log_validation(self, val_loss: float, batch: Dict):
        """Log validation results"""
        self.writer.add_scalar('validation/loss', val_loss, self.step)
        
        # Log sample spectrograms
        idx = 0  # First item in batch
        fig = plot_spectrogram(batch["mel_spectrogram"][idx],
                             title="Validation Ground Truth")
        self.writer.add_figure('validation/mel_target', fig, self.step)
        
        # Save model if best loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint('best')
            logger.info(f'New best model saved (loss: {val_loss:.4f})')
    
    def train(self, resume_checkpoint: Optional[str] = None):
        """Main training loop"""
        # Load checkpoint if specified
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
        
        try:
            # Training loop
            for epoch in range(self.epoch, self.max_epochs):
                self.epoch = epoch
                
                # Train epoch
                train_loss = self.train_epoch()
                logger.info(f'Epoch {epoch + 1}: train_loss = {train_loss:.4f}')
                
                # Validation
                val_loss = self.validate()
                logger.info(f'Epoch {epoch + 1}: val_loss = {val_loss:.4f}')
                
                # Save checkpoint
                self.save_checkpoint()
                
                # Save epoch checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'epoch_{epoch+1}')
        
        except KeyboardInterrupt:
            logger.info('Training interrupted by user')
            self.save_checkpoint('interrupted')
        
        except Exception as e:
            logger.error(f'Training failed: {str(e)}')
            self.save_checkpoint('failed')
            raise
        
        finally:
            self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train Stutter-TTS model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train(resume_checkpoint=args.resume)

if __name__ == "__main__":
    main()