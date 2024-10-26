import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .attention import MultiHeadAttention

class Prenet(nn.Module):
    """Decoder Prenet with Strong Regularization"""
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 dropout_rate: float = 0.6):
        """
        Args:
            in_dim: 入力次元（メルスペクトログラムのチャネル数）
            hidden_dim: 隠れ層の次元
            out_dim: 出力次元
            dropout_rate: ドロップアウト率（論文では0.6を使用）
        """
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        # 2層のフィードフォワードネットワーク
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        推論時もドロップアウトを適用する点が重要
        """
        # training modeに関係なくドロップアウトを適用
        return self.net(x)

class DecoderLayer(nn.Module):
    """Transformer Decoder Layer"""
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Self Attention
        self.self_attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout
        )
        
        # Encoder-Decoder Attention
        self.encoder_attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout
        )
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                self_attention_mask: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: デコーダー入力
            encoder_output: エンコーダー出力
            self_attention_mask: セルフアテンションのマスク
            encoder_attention_mask: エンコーダー-デコーダーアテンションのマスク
        """
        # Self Attention
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask=self_attention_mask)
        x = residual + self.dropout(x)
        
        # Encoder-Decoder Attention
        residual = x
        x = self.norm2(x)
        x = self.encoder_attention(x, encoder_output, encoder_output, mask=encoder_attention_mask)
        x = residual + self.dropout(x)
        
        # Feed Forward
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        
        return x

class Postnet(nn.Module):
    """Postnet for refining mel-spectrogram predictions"""
    def __init__(self,
                 mel_channels: int,
                 postnet_channels: int,
                 postnet_kernel_size: int,
                 postnet_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        # First Conv Layer
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(
                    mel_channels,
                    postnet_channels,
                    kernel_size=postnet_kernel_size,
                    padding=(postnet_kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(postnet_channels),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )
        
        # Hidden Conv Layers
        for _ in range(postnet_layers - 2):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        postnet_channels,
                        postnet_channels,
                        kernel_size=postnet_kernel_size,
                        padding=(postnet_kernel_size - 1) // 2
                    ),
                    nn.BatchNorm1d(postnet_channels),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
        
        # Last Conv Layer
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(
                    postnet_channels,
                    mel_channels,
                    kernel_size=postnet_kernel_size,
                    padding=(postnet_kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(mel_channels),
                nn.Dropout(dropout)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差接続として機能するPostnet出力を返す"""
        for conv in self.convs:
            x = conv(x)
        return x

class StopTokenPredictor(nn.Module):
    """Predicts when to stop mel-spectrogram generation"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    """Complete Decoder for Stutter-TTS"""
    def __init__(self,
                 mel_channels: int,
                 d_model: int,
                 n_head: int,
                 d_ff: int,
                 num_layers: int,
                 prenet_hidden_dim: int,
                 postnet_channels: int = 512,
                 postnet_kernel_size: int = 5,
                 postnet_layers: int = 5,
                 prenet_dropout: float = 0.6,
                 decoder_dropout: float = 0.1):
        super().__init__()
        
        # Prenet
        self.prenet = Prenet(
            in_dim=mel_channels,
            hidden_dim=prenet_hidden_dim,
            out_dim=d_model,
            dropout_rate=prenet_dropout
        )
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=decoder_dropout
            ) for _ in range(num_layers)
        ])
        
        # Output Projection
        self.mel_projection = nn.Linear(d_model, mel_channels)
        
        # Postnet
        self.postnet = Postnet(
            mel_channels=mel_channels,
            postnet_channels=postnet_channels,
            postnet_kernel_size=postnet_kernel_size,
            postnet_layers=postnet_layers,
            dropout=decoder_dropout
        )
        
        # Stop Token Predictor
        self.stop_predictor = StopTokenPredictor(d_model, decoder_dropout)
        
        # Final Layer Norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                encoder_output: torch.Tensor,
                mel_target: Optional[torch.Tensor] = None,
                encoder_padding_mask: Optional[torch.Tensor] = None,
                self_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_output: エンコーダーの出力
            mel_target: 教師強制用のメルスペクトログラム（学習時のみ）
            encoder_padding_mask: エンコーダー出力のパディングマスク
            self_attention_mask: デコーダーのセルフアテンションマスク
        Returns:
            mel_output: 予測メルスペクトログラム
            mel_postnet: Postnet適用後のメルスペクトログラム
            stop_tokens: 停止トークンの予測
        """
        # 学習時は教師強制、推論時は自己回帰
        if self.training and mel_target is not None:
            return self._training_forward(
                encoder_output,
                mel_target,
                encoder_padding_mask,
                self_attention_mask
            )
        else:
            return self._inference_forward(
                encoder_output,
                encoder_padding_mask
            )

    def _training_forward(self,
                         encoder_output: torch.Tensor,
                         mel_target: torch.Tensor,
                         encoder_padding_mask: Optional[torch.Tensor],
                         self_attention_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """学習時の順伝播"""
        # Prenetを通す（重要な正則化ポイント）
        x = self.prenet(mel_target)
        
        # デコーダーレイヤーの適用
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                self_attention_mask,
                encoder_padding_mask
            )
        
        x = self.norm(x)
        
        # メルスペクトログラムの生成
        mel_output = self.mel_projection(x)
        mel_postnet = mel_output + self.postnet(mel_output.transpose(1, 2)).transpose(1, 2)
        
        # 停止トークンの予測
        stop_tokens = self.stop_predictor(x)
        
        return mel_output, mel_postnet, stop_tokens

    def _inference_forward(self,
                          encoder_output: torch.Tensor,
                          encoder_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """推論時の自己回帰的生成"""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # 初期入力（ゼロで初期化）
        x = torch.zeros((batch_size, 1, self.mel_projection.out_features), device=device)
        
        mel_outputs = []
        stop_tokens = []
        
        # 自己回帰的生成
        max_length = 1000  # 最大生成長
        
        for _ in range(max_length):
            # Prenetを通す
            prenet_out = self.prenet(x[:, -1])
            prenet_out = prenet_out.unsqueeze(1)
            
            # デコーダーレイヤーの適用
            decoder_out = prenet_out
            for layer in self.layers:
                decoder_out = layer(
                    decoder_out,
                    encoder_output,
                    None,  # 推論時は不要
                    encoder_padding_mask
                )
            
            decoder_out = self.norm(decoder_out)
            
            # メルスペクトログラムと停止トークンの予測
            mel_frame = self.mel_projection(decoder_out)
            stop_token = self.stop_predictor(decoder_out)
            
            mel_outputs.append(mel_frame)
            stop_tokens.append(stop_token)
            
            # 停止判定
            if stop_token.item() > 0.5:
                break
            
            # 次のステップの入力
            x = torch.cat([x, mel_frame], dim=1)
        
        # 結果の結合
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_tokens = torch.cat(stop_tokens, dim=1)
        
        # Postnetの適用
        mel_postnet = mel_outputs + self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)
        
        return mel_outputs, mel_postnet, stop_tokens