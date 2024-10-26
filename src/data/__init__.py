from typing import Dict, Tuple
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import logging

from .audio_processor import AudioProcessor
from .text_processor import TextProcessor
from .dataset import (
    LibriSpeechDataset,
    create_dataloaders,
    collate_batch
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """設定ファイルの読み込み"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_paths(config: Dict) -> bool:
    """データセットのパスが有効かチェック"""
    librispeech_path = Path(config["data"]["librispeech_path"])
    
    if not librispeech_path.exists():
        logger.error(f"LibriSpeech path does not exist: {librispeech_path}")
        return False
        
    return True

def initialize_processors(config: Dict) -> Tuple[AudioProcessor, TextProcessor]:
    """音声とテキストの処理クラスを初期化"""
    audio_processor = AudioProcessor(config)
    text_processor = TextProcessor(config)
    return audio_processor, text_processor

def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """データローダーの作成"""
    logger.info("Creating data loaders...")
    
    try:
        train_loader, valid_loader, test_loader = create_dataloaders(config)
        
        logger.info(
            f"Created data loaders with batch size {config['training']['batch_size']}"
            f" and {config['training']['num_workers']} workers"
        )
        
        return train_loader, valid_loader, test_loader
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise

class StutterDataModule:
    """データ処理モジュールをまとめたクラス"""
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.audio_processor, self.text_processor = initialize_processors(self.config)
        
        # データローダーは遅延初期化
        self._train_loader = None
        self._valid_loader = None
        self._test_loader = None
        
    @property
    def train_loader(self) -> DataLoader:
        """訓練用データローダー"""
        if self._train_loader is None:
            self._initialize_loaders()
        return self._train_loader
        
    @property
    def valid_loader(self) -> DataLoader:
        """検証用データローダー"""
        if self._valid_loader is None:
            self._initialize_loaders()
        return self._valid_loader
        
    @property
    def test_loader(self) -> DataLoader:
        """テスト用データローダー"""
        if self._test_loader is None:
            self._initialize_loaders()
        return self._test_loader
    
    def _initialize_loaders(self):
        """データローダーの初期化"""
        loaders = create_data_loaders(self.config)
        self._train_loader, self._valid_loader, self._test_loader = loaders
    
    def process_single_text(self, text: str, stutter_events: Dict = None) -> Dict:
        """単一のテキストを処理"""
        return self.text_processor.process_text(text, stutter_events)
    
    def process_single_audio(self, audio_path: str, **kwargs) -> Dict:
        """単一の音声ファイルを処理"""
        return self.audio_processor.process_audio(audio_path, **kwargs)

def get_dataset_stats(config: Dict) -> Dict:
    """データセットの統計情報を取得"""
    train_dataset = LibriSpeechDataset(config, split="train")
    valid_dataset = LibriSpeechDataset(config, split="valid")
    test_dataset = LibriSpeechDataset(config, split="test")
    
    return {
        "train": {
            "total_samples": len(train_dataset),
        },
        "valid": {
            "total_samples": len(valid_dataset),
        },
        "test": {
            "total_samples": len(test_dataset),
        }
    }

# デフォルトのexport
__all__ = [
    'AudioProcessor',
    'TextProcessor',
    'LibriSpeechDataset',
    'create_dataloaders',
    'collate_batch',
    'StutterDataModule',
    'load_config',
    'get_dataset_stats'
]