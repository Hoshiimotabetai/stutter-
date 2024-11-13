import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from .attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置エンコーディングの計算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数インデックスにはsinを使用
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数インデックスにはcosを使用
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ProbabilisticPhoneticEncoder(nn.Module):
    """Probabilistic phonetic encoder with learnable variance"""
    def __init__(self, 
                 num_phonemes: int,
                 d_model: int,
                 dropout: float = 0.1,
                 padding_idx: int = 0):
        super().__init__()
        
        # エンベディングパラメータ
        self.phoneme_mu = nn.Parameter(torch.randn(num_phonemes, d_model))
        self.phoneme_logvar = nn.Parameter(torch.zeros(num_phonemes, d_model))
        
        # Position encoding weight
        self.alpha = nn.Parameter(torch.ones(1))
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Position encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Save padding index
        self.padding_idx = padding_idx
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize embedding parameters"""
        nn.init.xavier_normal_(self.phoneme_mu)
        nn.init.constant_(self.phoneme_logvar, -5.0)  # Start with small variance

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phoneme_ids: Phoneme ID tensor [batch_size, seq_len]
        Returns:
            Probabilistic phoneme embeddings [batch_size, seq_len, d_model]
        """
        # Create padding mask
        padding_mask = (phoneme_ids != self.padding_idx)
        padding_mask = padding_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Get embeddings
        mu = self.phoneme_mu[phoneme_ids]  # [batch, seq_len, d_model]
        logvar = self.phoneme_logvar[phoneme_ids]  # [batch, seq_len, d_model]
        
        # Apply reparameterization trick
        embeddings = self.reparameterize(mu, logvar)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding.pe[:, :embeddings.size(1)]
        embeddings = embeddings + self.alpha * pos_encoding
        
        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Apply padding mask
        embeddings = embeddings * padding_mask
        
        return embeddings

class EncoderLayer(nn.Module):
    """Transformer encoder layer"""
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout
        )
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
        """
        # Self attention
        att_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(att_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class GlobalAudioReferenceEncoder(nn.Module):
    """Global audio reference encoder for speaker embedding"""
    def __init__(self,
                 mel_channels: int = 80,
                 d_model: int = 512,
                 conv_channels: List[int] = [32, 32, 64, 64, 128, 128],
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        in_channels = mel_channels
        for out_channels in conv_channels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Number of frames to sample
        self.num_frames = 60

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: Mel spectrogram [batch_size, mel_channels, time]
        Returns:
            Global reference embedding [batch_size, d_model]
        """
        # Random frame sampling
        if mel_spec.size(2) > self.num_frames:
            indices = torch.randperm(mel_spec.size(2))[:self.num_frames]
            mel_spec = mel_spec[:, :, indices]
        
        # Apply convolutions
        x = mel_spec
        for conv in self.convs:
            x = conv(x)
        
        # Prepare for GRU
        x = x.transpose(1, 2)  # [batch, time, channels]
        
        # Apply GRU
        _, hidden = self.gru(x)
        reference_embedding = hidden[-1]  # Take last hidden state
        
        # Normalize output
        reference_embedding = self.layer_norm(reference_embedding)
        
        return reference_embedding

class Encoder(nn.Module):
    """Complete encoder for Stutter-TTS"""
    def __init__(self,
                 num_phonemes: int,
                 d_model: int,
                 n_head: int,
                 d_ff: int,
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Probabilistic phonetic encoder
        self.phonetic_encoder = ProbabilisticPhoneticEncoder(
            num_phonemes=num_phonemes,
            d_model=d_model,
            dropout=dropout
        )
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                phoneme_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phoneme_ids: Phoneme ID sequence [batch_size, seq_len]
            mask: Attention mask
        Returns:
            encoder_output: Encoded sequence
            phoneme_embeddings: Initial phoneme embeddings
        """
        # Get phoneme embeddings
        phoneme_embeddings = self.phonetic_encoder(phoneme_ids)
        x = self.dropout(phoneme_embeddings)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.layer_norm(x)
        
        return x, phoneme_embeddings

