import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """再現性のためにシードを設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイルの読み込み"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_dir: str, experiment_name: Optional[str] = None) -> str:
    """ロギングの設定"""
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = Path(log_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    
    return str(log_dir)

def save_fig(fig: plt.Figure, path: str):
    """図の保存"""
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_mel_spectrogram(mel_spec: torch.Tensor, title: str = '') -> plt.Figure:
    """メルスペクトログラムのプロット"""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mel_spec.detach().cpu().numpy(),
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig

def plot_attention(attention: torch.Tensor, title: str = '') -> plt.Figure:
    """アテンションマップのプロット"""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention.detach().cpu().numpy(),
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig

class LearningRateScheduler:
    """Transformer用の学習率スケジューラ"""
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        """学習率の更新"""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        """現在の学習率を計算"""
        step = self.current_step
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

class AverageMeter:
    """値の平均を追跡"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_training_state(state: Dict[str, Any], path: str):
    """トレーニング状態の保存"""
    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving training state: {str(e)}")

def load_training_state(path: str) -> Dict[str, Any]:
    """トレーニング状態の読み込み"""
    try:
        with open(path, 'r') as f:
            state = json.load(f)
        return state
    except Exception as e:
        logger.error(f"Error loading training state: {str(e)}")
        return {}

def get_gradient_norm(model: nn.Module) -> float:
    """モデルの勾配ノルムを計算"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def calculate_model_size(model: nn.Module) -> Dict[str, int]:
    """モデルのサイズを計算"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def create_experiment_directory(base_dir: str, experiment_name: Optional[str] = None) -> Dict[str, str]:
    """実験ディレクトリの作成"""
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = Path(base_dir) / experiment_name
    
    # サブディレクトリの作成
    dirs = {
        'checkpoints': exp_dir / 'checkpoints',
        'logs': exp_dir / 'logs',
        'samples': exp_dir / 'samples',
        'visualizations': exp_dir / 'visualizations'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in dirs.items()}