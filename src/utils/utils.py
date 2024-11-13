import torch
import numpy as np
import random
import yaml
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, Any, List

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_dir: str) -> None:
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def save_figure(fig: plt.Figure, path: str) -> None:
    """Save matplotlib figure"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_spectrogram(spectrogram: torch.Tensor, title: str = '') -> plt.Figure:
    """Plot mel spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram.detach().cpu().numpy(),
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig

def plot_attention(attention: torch.Tensor, title: str = '') -> plt.Figure:
    """Plot attention weights"""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention.detach().cpu().numpy(),
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig

class LearningRateScheduler:
    """Learning rate scheduler with warmup"""
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 d_model: int,
                 warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        """Calculate learning rate with warmup"""
        step = self.current_step
        return self.d_model ** (-0.5) * min(step ** (-0.5),
                                          step * self.warmup_steps ** (-1.5))

class AverageMeter:
    """Keep track of average values"""
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

def save_training_state(state: Dict[str, Any], path: str) -> None:
    """Save training state"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_training_state(path: str) -> Dict[str, Any]:
    """Load training state"""
    if not Path(path).exists():
        return {}
    return torch.load(path)

def calculate_gradient_norm(model: torch.nn.Module) -> float:
    """Calculate gradient norm of model parameters"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def create_experiment_directory(base_dir: str,
                              experiment_name: Optional[str] = None) -> Dict[str, str]:
    """Create experiment directory structure"""
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = Path(base_dir) / experiment_name
    
    # Create subdirectories
    dirs = {
        'root': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',
        'logs': exp_dir / 'logs',
        'samples': exp_dir / 'samples',
        'eval': exp_dir / 'eval'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in dirs.items()}