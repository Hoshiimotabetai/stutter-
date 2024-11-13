from typing import Dict, Optional, Tuple, List
import torch
import logging

from .attention import MultiHeadAttention
from .encoder import Encoder, GlobalAudioReferenceEncoder
from .decoder import Decoder
from .stutter_tts import StutterTTS, StutterTTSLoss

logger = logging.getLogger(__name__)

def create_model(config: Dict) -> StutterTTS:
    """Initialize StutterTTS model"""
    model = StutterTTS(config)
    logger.info(f"Created StutterTTS model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def create_optimizer(model: StutterTTS, config: Dict) -> torch.optim.Optimizer:
    """Initialize optimizer"""
    optimizer_config = config["training"]
    return torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
        eps=float(optimizer_config["epsilon"])
    )

def save_checkpoint(model: StutterTTS,
                   optimizer: torch.optim.Optimizer,
                   step: int,
                   epoch: int,
                   loss: float,
                   checkpoint_path: str):
    """Save model checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path: str,
                   model: StutterTTS,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint.get('step', 0)
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return step, epoch, loss

__all__ = [
    'StutterTTS',
    'StutterTTSLoss',
    'create_model',
    'create_optimizer',
    'save_checkpoint',
    'load_checkpoint',
    'MultiHeadAttention',
    'Encoder',
    'GlobalAudioReferenceEncoder',
    'Decoder'
]
