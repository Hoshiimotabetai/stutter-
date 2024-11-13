import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .attention import MultiHeadAttention

class Prenet(nn.Module):
    """Prenet with strong regularization to prevent exposure bias"""
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.dropout_rate = dropout_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Always apply dropout, even during inference"""
        x = F.dropout(F.relu(self.layer1(x)), p=self.dropout_rate, training=True)
        x = F.dropout(F.relu(self.layer2(x)), p=self.dropout_rate, training=True)
        return x

class PostNet(nn.Module):
    """Postnet to refine mel spectrogram prediction"""
    def __init__(self,
                 mel_channels: int,
                 postnet_channels: int,
                 postnet_kernel_size: int,
                 postnet_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # First conv layer
        self.convs = nn.ModuleList([
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
        ])
        
        # Hidden conv layers
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
        
        # Final conv layer
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
        """Return residual prediction"""
        for conv in self.convs:
            x = conv(x)
        return x

class DecoderLayer(nn.Module):
    """Transformer decoder layer"""
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Self attention
        self.self_attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout
        )
        
        # Encoder-decoder attention
        self.encoder_attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout
        )
        
        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
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
            x: Decoder input
            encoder_output: Encoder output
            self_attention_mask: Mask for self attention
            encoder_attention_mask: Mask for encoder-decoder attention
        """
        # Self attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x, mask=self_attention_mask)
        x = residual + self.dropout(x)
        
        # Encoder-decoder attention
        residual = x
        x = self.norm2(x)
        x, _ = self.encoder_attention(x, encoder_output, encoder_output,
                                    mask=encoder_attention_mask)
        x = residual + self.dropout(x)
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        
        return x

class StopTokenPredictor(nn.Module):
    """Predicts when to stop mel spectrogram generation"""
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
    """Complete decoder for Stutter-TTS"""
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
                 prenet_dropout: float = 0.5,
                 decoder_dropout: float = 0.1):
        super().__init__()
        
        # Prenet
        self.prenet = Prenet(
            in_dim=mel_channels,
            hidden_dim=prenet_hidden_dim,
            out_dim=d_model,
            dropout_rate=prenet_dropout
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=decoder_dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.mel_projection = nn.Linear(d_model, mel_channels)
        
        # Postnet
        self.postnet = PostNet(
            mel_channels=mel_channels,
            postnet_channels=postnet_channels,
            postnet_kernel_size=postnet_kernel_size,
            postnet_layers=postnet_layers,
            dropout=decoder_dropout
        )
        
        # Stop token predictor
        self.stop_predictor = StopTokenPredictor(d_model, decoder_dropout)
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                encoder_output: torch.Tensor,
                mel_target: Optional[torch.Tensor] = None,
                encoder_padding_mask: Optional[torch.Tensor] = None,
                self_attention_mask: Optional[torch.Tensor] = None,
                max_length: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_output: Output from encoder
            mel_target: Target mel spectrogram (for training)
            encoder_padding_mask: Mask for encoder attention
            self_attention_mask: Mask for decoder self attention
            max_length: Maximum generation length for inference
        Returns:
            mel_output: Generated mel spectrogram
            mel_postnet: Refined mel spectrogram
            stop_tokens: Stop token predictions
        """
        if mel_target is not None:
            # Training mode - teacher forcing
            # Pass target through prenet
            decoder_input = self.prenet(mel_target)
            
            # Apply decoder layers
            x = decoder_input
            for layer in self.layers:
                x = layer(
                    x,
                    encoder_output,
                    self_attention_mask,
                    encoder_padding_mask
                )
            
            # Final normalization
            x = self.norm(x)
            
            # Generate outputs
            mel_output = self.mel_projection(x)
            mel_postnet = mel_output + self.postnet(mel_output.transpose(1, 2)).transpose(1, 2)
            stop_tokens = self.stop_predictor(x)
            
            return mel_output, mel_postnet, stop_tokens
            
        else:
            # Inference mode - autoregressive generation
            device = encoder_output.device
            batch_size = encoder_output.size(0)
            
            # Initialize decoder input
            decoder_inputs = torch.zeros(
                (batch_size, 1, self.mel_projection.out_features),
                device=device
            )
            
            # Initialize outputs
            mel_outputs = []
            stop_tokens = []
            
            # Autoregressive loop
            for _ in range(max_length):
                # Pass through prenet
                x = self.prenet(decoder_inputs[:, -1:])
                
                # Apply decoder layers
                for layer in self.layers:
                    x = layer(
                        x,
                        encoder_output,
                        None,
                        encoder_padding_mask
                    )
                
                # Final normalization
                x = self.norm(x)
                
                # Generate next step
                mel_output = self.mel_projection(x)
                stop_token = self.stop_predictor(x)
                
                # Store outputs
                mel_outputs.append(mel_output)
                stop_tokens.append(stop_token)
                
                # Early stopping
                if stop_token.item() > 0.5:
                    break
                
                # Update decoder inputs
                decoder_inputs = torch.cat([decoder_inputs, mel_output], dim=1)
            
            # Concatenate outputs
            mel_outputs = torch.cat(mel_outputs, dim=1)
            stop_tokens = torch.cat(stop_tokens, dim=1)
            
            # Apply postnet
            mel_postnet = mel_outputs + self.postnet(
                mel_outputs.transpose(1, 2)
            ).transpose(1, 2)
            
            return mel_outputs, mel_postnet, stop_tokens

    def infer(self,
              encoder_output: torch.Tensor,
              encoder_padding_mask: Optional[torch.Tensor] = None,
              max_length: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference with maximum length limit"""
        return self.forward(
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            max_length=max_length
        )