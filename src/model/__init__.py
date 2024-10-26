from typing import Dict
import torch
import logging
from pathlib import Path

from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    RelativePositionMultiHeadAttention,
    create_padding_mask,
    create_look_ahead_mask,
    create_combined_mask
)

from .encoder import (
    PositionalEncoding,
    ProbabilisticPhoneticEncoder,
    EncoderLayer,
    Encoder,
    GlobalAudioReferenceEncoder
)

from .decoder import (
    Prenet,
    DecoderLayer,
    Postnet,
    StopTokenPredictor,
    Decoder
)

from .stutter_tts import (
    StutterTTS,
    StutterTTSLoss,
    inference,
    count_parameters,
    generate_square_subsequent_mask
)

logger = logging.getLogger(__name__)

def create_model(config: Dict) -> StutterTTS:
    """設定からStutterTTSモデルを作成"""
    try:
        model = StutterTTS(config)
        logger.info(f"Created StutterTTS model with {count_parameters(model):,} parameters")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def create_optimizer(model: StutterTTS, config: Dict) -> torch.optim.Optimizer:
    """モデルの最適化器を作成"""
    optimizer_config = config["training"]["optimizer"]
    optimizer_name = optimizer_config.get("name", "Adam").lower()
    learning_rate = optimizer_config.get("learning_rate", 1e-4)
    
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(float(optimizer_config.get("beta1", 0.9)),
                   float(optimizer_config.get("beta2", 0.998))),
            eps=float(optimizer_config.get("epsilon", 1e-9)),
            weight_decay=float(optimizer_config.get("weight_decay", 1e-6))
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer

def load_checkpoint(model: StutterTTS,
                   optimizer: torch.optim.Optimizer,
                   checkpoint_path: str) -> tuple:
    """チェックポイントの読み込み"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        return 0, 0
    
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        
        logger.info(f"Loaded checkpoint from epoch {epoch} (global step: {global_step})")
        return epoch, global_step
    
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise

def save_checkpoint(model: StutterTTS,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   global_step: int,
                   checkpoint_dir: str,
                   name: str = "checkpoint") -> None:
    """チェックポイントの保存"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{name}_epoch{epoch}.pt"
    
    try:
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise

# デフォルトのexport
__all__ = [
    # モデルとロス
    'StutterTTS',
    'StutterTTSLoss',
    
    # Attention関連
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'RelativePositionMultiHeadAttention',
    
    # エンコーダー関連
    'PositionalEncoding',
    'ProbabilisticPhoneticEncoder',
    'EncoderLayer',
    'Encoder',
    'GlobalAudioReferenceEncoder',
    
    # デコーダー関連
    'Prenet',
    'DecoderLayer',
    'Postnet',
    'StopTokenPredictor',
    'Decoder',
    
    # ユーティリティ関数
    'create_model',
    'create_optimizer',
    'load_checkpoint',
    'save_checkpoint',
    'inference',
    'count_parameters',
    
    # マスク関連
    'create_padding_mask',
    'create_look_ahead_mask',
    'create_combined_mask',
    'generate_square_subsequent_mask'
]

# バージョン情報
__version__ = '1.0.0'
