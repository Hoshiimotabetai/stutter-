from typing import Dict, Tuple
from torch.utils.data import DataLoader
import logging

from .audio_processor import AudioProcessor
from .text_processor import TextProcessor
from .dataset import LibriSpeechDataset, collate_batch

logger = logging.getLogger(__name__)

def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    logger.info("Creating data loaders...")
    
    # Create datasets
    train_dataset = LibriSpeechDataset(config, split="train")
    valid_dataset = LibriSpeechDataset(config, split="valid")
    test_dataset = LibriSpeechDataset(config, split="test")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_batch,
        pin_memory=config["training"].get("pin_memory", True),
        prefetch_factor=config["training"].get("prefetch_factor", 2)
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_batch,
        pin_memory=config["training"].get("pin_memory", True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_batch,
        pin_memory=config["training"].get("pin_memory", True)
    )
    
    logger.info(f"Created data loaders with batch size {config['training']['batch_size']}")
    
    return train_loader, valid_loader, test_loader

__all__ = [
    'AudioProcessor',
    'TextProcessor',
    'LibriSpeechDataset',
    'create_dataloaders',
    'collate_batch'
]