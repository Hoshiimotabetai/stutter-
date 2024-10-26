import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor

class LibriSpeechDataset(Dataset):
    """LibriSpeechデータセット用のクラス"""
    def __init__(self, config, split="train"):
        """
        Args:
            config: 設定ファイルから読み込んだ設定辞書
            split: データセットの分割（'train', 'valid', 'test'）
        """
        self.config = config
        self.split = split
        self.audio_processor = AudioProcessor(config)
        self.text_processor = TextProcessor(config)
        
        # データの読み込み
        self.data = self._load_librispeech()
        
        print(f"[{split}] Loaded {len(self.data)} utterances")

    def _load_librispeech(self) -> List[Dict]:
        """LibriSpeechデータセットの読み込み"""
        data = []
        librispeech_path = Path(self.config["data"]["librispeech_path"])
        
        # 分割に応じたサブセットの設定
        if self.split == "train":
            subsets = ["train-clean-100"]
        elif self.split == "valid":
            subsets = ["dev-clean"]
        else:  # test
            subsets = ["test-clean"]
        
        for subset in subsets:
            subset_path = librispeech_path / subset
            
            for trans_file in subset_path.glob("**/*.trans.txt"):
                chapter_dir = trans_file.parent
                with open(trans_file, "r", encoding="utf-8") as f:
                    for line in f:
                        file_id, text = line.strip().split(" ", 1)
                        wav_path = chapter_dir / f"{file_id}.flac"
                        
                        if wav_path.exists():
                            data.append({
                                "utterance_id": file_id,
                                "speaker_id": chapter_dir.parent.name,
                                "text": text,
                                "audio_path": str(wav_path),
                                "is_stutter": False,
                                "stutter_events": None
                            })
        
        return data

    def _process_item(self, item: Dict) -> Dict:
        """データ項目の処理"""
        # 音声の処理
        audio_data = self.audio_processor.process_audio(
            item["audio_path"],
            max_duration=self.config["data"].get("max_duration")
        )
        
        # テキストの処理
        text_data = self.text_processor.process_text(
            item["text"]
        )
        
        return {
            "utterance_id": item["utterance_id"],
            "speaker_id": item["speaker_id"],
            "text": item["text"],
            "waveform": audio_data["waveform"],
            "mel_spectrogram": audio_data["mel_spectrogram"],
            "duration": audio_data["duration"],
            "phoneme_ids": torch.LongTensor(text_data["phoneme_ids"]),
            "is_stutter": False,  # LibriSpeechには吃音データがない
            "stutter_events": None
        }

    def __len__(self) -> int:
        """データセットの長さを返す"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """インデックスに対応するデータ項目を返す"""
        return self._process_item(self.data[idx])

def collate_batch(batch: List[Dict]) -> Dict:
    """バッチデータのパディングと処理"""
    # バッチ内の最大長を取得
    max_waveform_len = max(x["waveform"].size(1) for x in batch)
    max_mel_len = max(x["mel_spectrogram"].size(2) for x in batch)
    max_phoneme_len = max(x["phoneme_ids"].size(0) for x in batch)
    
    # バッチサイズ
    batch_size = len(batch)
    
    # パディング済みテンソルの準備
    waveform_padded = torch.zeros(batch_size, 1, max_waveform_len)
    mel_padded = torch.zeros(batch_size, 80, max_mel_len)  # 80はメル周波数ビンの数
    phoneme_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    
    # マスクの準備
    waveform_masks = torch.zeros(batch_size, max_waveform_len)
    mel_masks = torch.zeros(batch_size, max_mel_len)
    phoneme_masks = torch.zeros(batch_size, max_phoneme_len)
    
    # メタデータを格納するリスト
    metadata = {
        "utterance_ids": [],
        "speaker_ids": [],
        "texts": [],
        "durations": [],
        "is_stutter": [],
        "stutter_events": []
    }
    
    for i, item in enumerate(batch):
        # 波形のパディング
        waveform = item["waveform"]
        waveform_len = waveform.size(1)
        waveform_padded[i, :, :waveform_len] = waveform
        waveform_masks[i, :waveform_len] = 1
        
        # メルスペクトログラムのパディング
        mel_spec = item["mel_spectrogram"]
        mel_len = mel_spec.size(2)
        mel_padded[i, :, :mel_len] = mel_spec
        mel_masks[i, :mel_len] = 1
        
        # 音素IDのパディング
        phoneme_ids = item["phoneme_ids"]
        phoneme_len = phoneme_ids.size(0)
        phoneme_padded[i, :phoneme_len] = phoneme_ids
        phoneme_masks[i, :phoneme_len] = 1
        
        # メタデータの追加
        metadata["utterance_ids"].append(item["utterance_id"])
        metadata["speaker_ids"].append(item["speaker_id"])
        metadata["texts"].append(item["text"])
        metadata["durations"].append(item["duration"])
        metadata["is_stutter"].append(item["is_stutter"])
        metadata["stutter_events"].append(item["stutter_events"])
    
    return {
        **metadata,
        "waveform": waveform_padded,
        "mel_spectrogram": mel_padded,
        "phoneme_ids": phoneme_padded,
        "waveform_masks": waveform_masks,
        "mel_masks": mel_masks,
        "phoneme_masks": phoneme_masks,
        "is_stutter": torch.tensor([False] * batch_size, dtype=torch.bool)
    }

def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """トレーニング、検証、テスト用のデータローダーを作成"""
    
    # データセットの作成
    train_dataset = LibriSpeechDataset(config, split="train")
    valid_dataset = LibriSpeechDataset(config, split="valid")
    test_dataset = LibriSpeechDataset(config, split="test")
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_batch
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_batch
    )
    
    return train_loader, valid_loader, test_loader