import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import os
import random
import numpy as np

from model import (
    StutterTTS,
    StutterTTSLoss,
    create_model,
    create_optimizer,
    load_checkpoint,
    save_checkpoint
)
from data import create_dataloaders
from utils import (
    set_seed,
    load_config,
    setup_logging,
    LearningRateScheduler,
    AverageMeter,
    plot_mel_spectrogram,
    plot_attention,
    save_training_state,
    load_training_state,
    get_gradient_norm,
    calculate_model_size,
    create_experiment_directory
)

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: Dict, experiment_dir: Optional[str] = None):
        self.config = config
        
        # デバイスの設定
        self.device = torch.device(config["training"]["device"])
        
        # 実験ディレクトリの設定
        if experiment_dir is None:
            dirs = create_experiment_directory(
                config["training"]["experiment_dir"]
            )
        self.experiment_dir = dirs
        
        # ロギングの設定
        setup_logging(self.experiment_dir["logs"])
        self.writer = SummaryWriter(self.experiment_dir["logs"])
        
        # モデルの作成
        self.model = create_model(config).to(self.device)
        model_size = calculate_model_size(self.model)
        logger.info(f"Model size: {model_size}")
        
        # 損失関数
        self.criterion = StutterTTSLoss()
        
        # 最適化器
        self.optimizer = create_optimizer(self.model, config)
        
        # スケジューラ
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            config["model"]["d_model"],
            config["training"]["warmup_steps"]
        )
        
        # データローダー
        self.train_loader, self.valid_loader, self.test_loader = create_dataloaders(config)
        
        # トレーニング状態
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """チェックポイントの読み込み"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.experiment_dir["checkpoints"],
                "latest.pt"
            )
        
        if os.path.exists(checkpoint_path):
            self.epoch, self.global_step = load_checkpoint(
                self.model,
                self.optimizer,
                checkpoint_path
            )
            
            # トレーニング状態の読み込み
            state_path = Path(checkpoint_path).with_suffix('.json')
            if state_path.exists():
                state = load_training_state(str(state_path))
                self.best_loss = state.get('best_loss', float('inf'))
                logger.info(f"Resumed from epoch {self.epoch} (global step: {self.global_step})")
    
    def save_checkpoint(self, name: str = "latest"):
        """チェックポイントの保存"""
        save_checkpoint(
            self.model,
            self.optimizer,
            self.epoch,
            self.global_step,
            self.experiment_dir["checkpoints"],
            name
        )
        
        # トレーニング状態の保存
        state_path = os.path.join(
            self.experiment_dir["checkpoints"],
            f"{name}_epoch{self.epoch}.json"
        )
        save_training_state(
            {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss
            },
            state_path
        )
    
    def train_epoch(self) -> float:
        """1エポックの学習"""
        self.model.train()
        epoch_loss = AverageMeter()
        
        with tqdm(total=len(self.train_loader), desc=f'Epoch {self.epoch + 1}') as pbar:
            for batch in self.train_loader:
                # データをデバイスに移動
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # フォワードパス
                mel_output, mel_postnet, stop_tokens = self.model(
                    phoneme_ids=batch["phoneme_ids"],
                    mel_target=batch["mel_spectrogram"],
                    reference_mel=batch.get("reference_mel")
                )
                
                # 損失の計算
                loss, loss_dict = self.criterion(
                    mel_output,
                    mel_postnet,
                    stop_tokens,
                    batch["mel_spectrogram"],
                    batch["mel_masks"]
                )
                
                # バックワードパス
                self.optimizer.zero_grad()
                loss.backward()
                
                # 勾配クリッピング
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["grad_clip_thresh"]
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # メトリクスの更新
                epoch_loss.update(loss.item())
                self.global_step += 1
                
                # ログの記録
                if self.global_step % self.config["training"]["log_interval"] == 0:
                    self._log_training_step(loss_dict, batch, grad_norm)
                
                # プログレスバーの更新
                pbar.set_postfix(loss=f'{epoch_loss.avg:.4f}', lr=f'{self.scheduler._get_lr():.6f}')
                pbar.update(1)
        
        return epoch_loss.avg
    
    @torch.no_grad()
    def validate(self) -> float:
        """検証"""
        self.model.eval()
        val_loss = AverageMeter()
        
        for batch in tqdm(self.valid_loader, desc='Validation'):
            # データをデバイスに移動
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # フォワードパス
            mel_output, mel_postnet, stop_tokens = self.model(
                phoneme_ids=batch["phoneme_ids"],
                mel_target=batch["mel_spectrogram"],
                reference_mel=batch.get("reference_mel")
            )
            
            # 損失の計算
            loss, loss_dict = self.criterion(
                mel_output,
                mel_postnet,
                stop_tokens,
                batch["mel_spectrogram"],
                batch["mel_masks"]
            )
            
            val_loss.update(loss.item())
        
        # バリデーションの結果をログ
        self._log_validation(val_loss.avg, batch)
        
        return val_loss.avg
    
    def _log_training_step(self, loss_dict: Dict[str, float], batch: Dict, grad_norm: float):
        """トレーニングステップのログ記録"""
        # 損失のログ
        for name, value in loss_dict.items():
            self.writer.add_scalar(f'train/{name}', value, self.global_step)
        
        # 勾配ノルムのログ
        self.writer.add_scalar('train/gradient_norm', grad_norm, self.global_step)
        
        # 学習率のログ
        self.writer.add_scalar('train/learning_rate', self.scheduler._get_lr(), self.global_step)
        
        # サンプルの可視化（定期的に）
        if self.global_step % self.config["training"]["sample_interval"] == 0:
            self._log_samples(batch, prefix='train')
    
    def _log_validation(self, val_loss: float, batch: Dict):
        """バリデーション結果のログ記録"""
        self.writer.add_scalar('validation/loss', val_loss, self.global_step)
        
        # サンプルの可視化
        self._log_samples(batch, prefix='validation')
        
        # 最良モデルの保存
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint('best')
            logger.info(f'New best model saved (loss: {val_loss:.4f})')
    
    def _log_samples(self, batch: Dict, prefix: str = 'train'):
        """生成サンプルの可視化"""
        idx = random.randint(0, len(batch["mel_spectrogram"]) - 1)
        
        # 真のメルスペクトログラム
        fig_true = plot_mel_spectrogram(
            batch["mel_spectrogram"][idx],
            title='Ground Truth Mel-spectrogram'
        )
        self.writer.add_figure(f'{prefix}/true_mel', fig_true, self.global_step)
        
        # 生成されたメルスペクトログラム
        with torch.no_grad():
            mel_output, mel_postnet, _ = self.model(
                phoneme_ids=batch["phoneme_ids"][idx:idx+1],
                reference_mel=batch.get("reference_mel")[idx:idx+1] if batch.get("reference_mel") is not None else None
            )
        
        fig_pred = plot_mel_spectrogram(
            mel_postnet[0],
            title='Generated Mel-spectrogram'
        )
        self.writer.add_figure(f'{prefix}/predicted_mel', fig_pred, self.global_step)
        
        # サンプルの保存
        if self.global_step % self.config["training"]["save_sample_interval"] == 0:
            sample_dir = Path(self.experiment_dir["samples"]) / f'step_{self.global_step}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            fig_true.savefig(sample_dir / f'{prefix}_true_mel.png')
            fig_pred.savefig(sample_dir / f'{prefix}_predicted_mel.png')
    
    def train(self, resume_checkpoint: Optional[str] = None):
        """トレーニングのメインループ"""
        # チェックポイントからの復帰
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
        
        try:
            # トレーニングループ
            for epoch in range(self.epoch, self.config["training"]["epochs"]):
                self.epoch = epoch
                
                # トレーニング
                train_loss = self.train_epoch()
                logger.info(f'Epoch {epoch + 1}: train_loss = {train_loss:.4f}')
                
                # 検証
                val_loss = self.validate()
                logger.info(f'Epoch {epoch + 1}: val_loss = {val_loss:.4f}')
                
                # チェックポイントの保存
                self.save_checkpoint('latest')
                
                # 定期的なチェックポイント
                if (epoch + 1) % self.config["training"]["checkpoint_interval"] == 0:
                    self.save_checkpoint(f'epoch_{epoch + 1}')
        
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
    
    # シードの設定
    set_seed(args.seed)
    
    # 設定の読み込み
    config = load_config(args.config)
    
    # トレーナーの作成と学習の実行
    trainer = Trainer(config)
    trainer.train(resume_checkpoint=args.resume)

if __name__ == "__main__":
    main()
