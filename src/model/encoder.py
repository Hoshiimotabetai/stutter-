

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from .attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    """位置エンコーディング"""
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 位置エンコーディングの計算
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数インデックスにはsinを使用
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数インデックスにはcosを使用
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # バッチ次元の追加
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル [batch_size, seq_length, d_model]
        Returns:
            位置情報が追加されたテンソル
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ProbabilisticPhoneticEncoder(nn.Module):
    """確率的音素エンコーダー"""
    def __init__(self, 
                 num_phonemes: int,
                 d_model: int,
                 dropout: float = 0.1,
                 padding_idx: int = 0):
        super().__init__()
        
        # 音素埋め込みの平均と分散のパラメータ
        self.phoneme_mu = nn.Parameter(torch.randn(num_phonemes, d_model))
        self.phoneme_logvar = nn.Parameter(torch.zeros(num_phonemes, d_model))
        
        # パディングインデックスの設定
        self.padding_idx = padding_idx
        
        # 位置エンコーディングの重み
        self.alpha = nn.Parameter(torch.ones(1))
        
        # 正規化とドロップアウト
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # 位置エンコーディング
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # パラメータの初期化
        nn.init.xavier_normal_(self.phoneme_mu)
        nn.init.constant_(self.phoneme_logvar, -5.0)  # 小さな初期分散

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phoneme_ids: 音素IDのテンソル [batch_size, seq_length]
        Returns:
            確率的な音素埋め込み [batch_size, seq_length, d_model]
        """
        # パディングマスクの作成
        padding_mask = (phoneme_ids != self.padding_idx)
        
        # 音素埋め込みの取得
        mu = self.phoneme_mu[phoneme_ids]  # [batch, seq_len, d_model]
        logvar = self.phoneme_logvar[phoneme_ids]  # [batch, seq_len, d_model]
        
        # 確率的埋め込みの生成
        embeddings = self.reparameterize(mu, logvar)
        
        # 位置エンコーディングの適用
        pos_encoding = self.positional_encoding.pe[:, :embeddings.size(1)]
        embeddings = embeddings + self.alpha * pos_encoding
        
        # 正規化とドロップアウト
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # パディングマスクの適用
        embeddings = embeddings * padding_mask.unsqueeze(-1)
        
        return embeddings

class FeedForward(nn.Module):
    """フィードフォワードネットワーク"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(self.linear2(x))
        x = self.layer_norm(x + residual)
        return x

class EncoderLayer(nn.Module):
    """Transformerエンコーダーレイヤー"""
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_head, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self Attention
        att_output = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(att_output))
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    """完全なエンコーダー"""
    def __init__(self,
                 num_phonemes: int,
                 d_model: int,
                 n_head: int,
                 d_ff: int,
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # 確率的音素エンコーダー
        self.phonetic_encoder = ProbabilisticPhoneticEncoder(
            num_phonemes=num_phonemes,
            d_model=d_model,
            dropout=dropout
        )
        
        # エンコーダーレイヤーのスタック
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 出力の正規化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, 
                phoneme_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phoneme_ids: 音素IDのテンソル [batch_size, seq_length]
            mask: アテンションマスク（オプション）
        Returns:
            エンコーダー出力と音素埋め込み
        """
        # 確率的音素エンコーディング
        phoneme_embeddings = self.phonetic_encoder(phoneme_ids)
        
        # エンコーダーレイヤーの適用
        encoder_output = phoneme_embeddings
        for layer in self.layers:
            encoder_output = layer(encoder_output, mask)
        
        # 最終正規化
        encoder_output = self.layer_norm(encoder_output)
        
        return encoder_output, phoneme_embeddings

class GlobalAudioReferenceEncoder(nn.Module):
    """グローバルオーディオリファレンスエンコーダー"""
    def __init__(self, 
                 mel_channels: int = 80,
                 d_model: int = 512,
                 conv_channels: List[int] = [32, 32, 64, 64, 128, 128],
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        # 畳み込み層
        self.convs = nn.ModuleList()
        in_channels = mel_channels
        for out_channels in conv_channels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # GRU
        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=d_model,
            batch_first=True
        )
        
        # 出力の正規化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # ランダムサンプリングのフレーム数
        self.num_frames = 60

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: メルスペクトログラム [batch_size, mel_channels, time]
        Returns:
            リファレンス埋め込み [batch_size, d_model]
        """
        # ランダムフレームのサンプリング
        if mel_spec.size(2) > self.num_frames:
            indices = torch.randperm(mel_spec.size(2))[:self.num_frames]
            mel_spec = mel_spec[:, :, indices]
        
        # 畳み込み層の適用
        x = mel_spec
        for conv in self.convs:
            x = conv(x)
        
        # GRUの入力形式に変換
        x = x.transpose(1, 2)  # [batch, time, channels]
        
        # GRUの適用
        _, hidden = self.gru(x)
        reference_embedding = hidden[-1]  # 最後の隠れ状態を使用
        
        # 正規化
        reference_embedding = self.layer_norm(reference_embedding)
        
        return reference_embedding