import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config["data"]["sample_rate"]
        self.n_mels = config["data"]["n_mels"]
        self.n_fft = config["data"]["n_fft"]
        self.hop_length = config["data"]["hop_length"]
        self.win_length = config["data"]["win_length"]
        
        # メルスペクトログラム変換の初期化
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=1.0,
            normalized=True,
            center=True,
            pad_mode='reflect',
            norm='slaney'
        )
        
        # 正規化用のスケーリング係数
        self.mel_scale_factor = 20.0
        
        # エネルギー正規化のパラメータ
        self.ref_level_db = 20
        self.min_level_db = -100

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """音声ファイルの読み込み（エラーハンドリング付き）"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            return waveform, sr
        except Exception as e:
            warnings.warn(f"Error loading audio file {audio_path}: {str(e)}")
            raise

    def resample_if_necessary(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """サンプリングレートの変換"""
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interpolation",
                beta=None
            )
            waveform = resampler(waveform)
        return waveform

    def convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """ステレオからモノラルへの変換"""
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def normalize_volume(self, waveform: torch.Tensor) -> torch.Tensor:
        """音量の正規化"""
        norm_factor = torch.abs(waveform).max()
        if norm_factor > 0:
            waveform = waveform / norm_factor
        return waveform

    def trim_silence(self, waveform: torch.Tensor, threshold_db: float = 40) -> torch.Tensor:
        """無音区間の除去（音声認識用のロバストな実装）"""
        waveform_np = waveform.numpy().squeeze()
        
        # トリミング
        yt, _ = librosa.effects.trim(
            waveform_np,
            top_db=threshold_db,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )
        
        return torch.from_numpy(yt).unsqueeze(0)

    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """メルスペクトログラムの計算"""
        # メルスペクトログラムの計算
        mel_spec = self.mel_transform(waveform)
        
        # 対数変換と正規化
        mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-5))
        mel_spec = torch.clamp(
            (mel_spec - self.min_level_db) / (-self.min_level_db),
            0.0, 1.0
        )
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        return mel_spec

    def process_audio(self, 
                     audio_path: str, 
                     max_duration: Optional[float] = None,
                     trim_silence: bool = True) -> Dict:
        """音声の前処理を一括で実行"""
        try:
            # 音声の読み込み
            waveform, sr = self.load_audio(audio_path)
            
            # 基本的な前処理
            waveform = self.convert_to_mono(waveform)
            waveform = self.resample_if_necessary(waveform, sr)
            waveform = self.normalize_volume(waveform)
            
            # 無音除去（オプション）
            if trim_silence:
                waveform = self.trim_silence(waveform)
            
            # 最大長でカット
            if max_duration is not None:
                max_samples = int(max_duration * self.sample_rate)
                if waveform.size(1) > max_samples:
                    waveform = waveform[:, :max_samples]
            
            # メルスペクトログラムの計算
            mel_spec = self.get_mel_spectrogram(waveform)
            
            return {
                "waveform": waveform,
                "mel_spectrogram": mel_spec,
                "duration": waveform.size(1) / self.sample_rate,
                "sample_rate": self.sample_rate
            }
            
        except Exception as e:
            warnings.warn(f"Error processing audio file {audio_path}: {str(e)}")
            raise

    def save_audio(self, 
                  waveform: torch.Tensor, 
                  file_path: str,
                  sample_rate: Optional[int] = None) -> None:
        """音声の保存（デバッグ用）"""
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        torchaudio.save(
            file_path,
            waveform,
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )