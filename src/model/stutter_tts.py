import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from .encoder import Encoder, GlobalAudioReferenceEncoder
from .decoder import Decoder

class StutterTTS(nn.Module):
    """Complete Stutter-TTS model"""
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        model_config = config["model"]
        self.d_model = model_config["d_model"]
        self.n_mels = config["data"]["n_mels"]
        
        # Encoder (includes probabilistic phonetic encoder)
        self.encoder = Encoder(
            num_phonemes=model_config["n_phonemes"],
            d_model=self.d_model,
            n_head=model_config["n_heads"],
            d_ff=model_config["d_ff"],
            num_layers=model_config["n_encoder_layers"],
            dropout=model_config["dropout"]
        )
        
        # Global Reference Encoder for speaker embedding
        self.reference_encoder = GlobalAudioReferenceEncoder(
            mel_channels=self.n_mels,
            d_model=self.d_model,
            conv_channels=model_config.get("ref_enc_filters", [32, 32, 64, 64, 128, 128]),
            kernel_size=model_config.get("ref_enc_kernel_size", 3),
            dropout=model_config["dropout"]
        )
        
        # Decoder
        self.decoder = Decoder(
            mel_channels=self.n_mels,
            d_model=self.d_model,
            n_head=model_config["n_heads"],
            d_ff=model_config["d_ff"],
            num_layers=model_config["n_decoder_layers"],
            prenet_hidden_dim=self.d_model // 2,
            postnet_channels=model_config["postnet_channels"],
            postnet_kernel_size=model_config["postnet_kernel_size"],
            postnet_layers=model_config.get("postnet_layers", 5),
            prenet_dropout=model_config["prenet_dropout"],
            decoder_dropout=model_config["dropout"]
        )
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_masks(self,
                     src_seq: torch.Tensor,
                     tgt_seq: Optional[torch.Tensor] = None,
                     src_pad_idx: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create masks for encoder and decoder"""
        # Source padding mask [batch_size, 1, 1, src_len]
        src_mask = (src_seq != src_pad_idx).unsqueeze(1).unsqueeze(2).to(torch.float32)
        
        # Target mask for decoder (if in training)
        if tgt_seq is not None:
            tgt_len = tgt_seq.size(2)  # [batch_size, n_mels, time]
            
            # Create padding mask [batch_size, 1, time, time]
            tgt_pad_mask = torch.ones(
                (tgt_seq.size(0), 1, tgt_len, tgt_len),
                device=tgt_seq.device,
                dtype=torch.float32
            )
            
            # Create causal mask [1, time, time]
            subsequent_mask = torch.triu(
                torch.ones((1, tgt_len, tgt_len), device=tgt_seq.device),
                diagonal=1
            )
            
            # Combine masks
            tgt_mask = tgt_pad_mask * (1.0 - subsequent_mask)
        else:
            tgt_mask = None
            
        return src_mask, tgt_mask

    def forward(self,
                phoneme_ids: torch.Tensor,
                mel_target: Optional[torch.Tensor] = None,
                reference_mel: Optional[torch.Tensor] = None,
                max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            phoneme_ids: phoneme ID sequence [batch_size, text_len]
            mel_target: target mel spectrogram [batch_size, n_mels, mel_len]
            reference_mel: reference audio [batch_size, n_mels, ref_len]
            max_length: maximum generation length
        Returns:
            mel_output: generated mel spectrogram
            mel_postnet: post-processed mel spectrogram
            stop_tokens: stop token predictions
        """
        # Create masks
        src_mask, tgt_mask = self._create_masks(phoneme_ids, mel_target)
        
        # Encoder
        encoder_output, phoneme_embeddings = self.encoder(phoneme_ids, src_mask)
        
        # Reference encoder (if provided)
        if reference_mel is not None:
            ref_embedding = self.reference_encoder(reference_mel)
            encoder_output = encoder_output + ref_embedding.unsqueeze(1)
        
        # Decoder
        if mel_target is not None:
            # Training mode
            mel_input = mel_target.transpose(1, 2)  # [batch, time, mel]
            mel_output, mel_postnet, stop_tokens = self.decoder(
                encoder_output=encoder_output,
                mel_target=mel_input,
                encoder_padding_mask=src_mask,
                self_attention_mask=tgt_mask
            )
            
            # Convert back to [batch, mel, time]
            mel_output = mel_output.transpose(1, 2)
            mel_postnet = mel_postnet.transpose(1, 2)
            
        else:
            # Inference mode
            if max_length is None:
                max_length = encoder_output.size(1) * 10
                
            mel_output, mel_postnet, stop_tokens = self.decoder.infer(
                encoder_output=encoder_output,
                encoder_padding_mask=src_mask,
                max_length=max_length
            )
        
        return mel_output, mel_postnet, stop_tokens

    def inference(self,
                 phoneme_ids: torch.Tensor,
                 reference_mel: Optional[torch.Tensor] = None,
                 max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode
        Args:
            phoneme_ids: phoneme ID sequence [batch_size, text_len]
            reference_mel: reference audio
            max_length: maximum generation length
        Returns:
            mel_postnet: generated mel spectrogram
            stop_tokens: stop token predictions
        """
        self.eval()
        with torch.no_grad():
            _, mel_postnet, stop_tokens = self.forward(
                phoneme_ids=phoneme_ids,
                reference_mel=reference_mel,
                max_length=max_length
            )
        return mel_postnet, stop_tokens

    def get_attention_weights(self) -> Dict[str, List[torch.Tensor]]:
        """Get attention weights for visualization"""
        attention_weights = {
            "encoder_self_attention": [],
            "decoder_self_attention": [],
            "encoder_decoder_attention": []
        }
        
        # Collect encoder self attention
        for layer in self.encoder.layers:
            if hasattr(layer.self_attention, 'attention'):
                attention_weights["encoder_self_attention"].append(
                    layer.self_attention.attention.detach()
                )
        
        # Collect decoder attentions
        for layer in self.decoder.layers:
            if hasattr(layer.self_attention, 'attention'):
                attention_weights["decoder_self_attention"].append(
                    layer.self_attention.attention.detach()
                )
            if hasattr(layer.encoder_attention, 'attention'):
                attention_weights["encoder_decoder_attention"].append(
                    layer.encoder_attention.attention.detach()
                )
        
        return attention_weights

class StutterTTSLoss(nn.Module):
    """Loss function for Stutter-TTS"""
    def __init__(self, config: Dict):
        super().__init__()
        self.mel_loss_weight = config["loss"]["mel_loss_weight"]
        self.postnet_mel_loss_weight = config["loss"]["postnet_mel_loss_weight"]
        self.stop_token_loss_weight = config["loss"]["stop_token_loss_weight"]

    def forward(self,
                mel_output: torch.Tensor,
                mel_postnet: torch.Tensor,
                stop_tokens: torch.Tensor,
                mel_target: torch.Tensor,
                stop_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate loss
        Args:
            mel_output: decoder output
            mel_postnet: postnet output
            stop_tokens: stop token predictions
            mel_target: target mel spectrogram
            stop_target: target stop tokens
        Returns:
            total_loss: combined loss value
            loss_dict: dictionary containing individual losses
        """
        # Mel spectrogram L1 losses
        mel_loss = F.l1_loss(mel_output, mel_target)
        postnet_mel_loss = F.l1_loss(mel_postnet, mel_target)
        
        # Stop token binary cross entropy loss
        stop_loss = F.binary_cross_entropy_with_logits(
            stop_tokens.squeeze(-1),
            stop_target.float()
        )
        
        # Combine losses with weights
        total_loss = (
            self.mel_loss_weight * mel_loss +
            self.postnet_mel_loss_weight * postnet_mel_loss +
            self.stop_token_loss_weight * stop_loss
        )
        
        # Create loss dictionary
        loss_dict = {
            'mel_loss': mel_loss.item(),
            'postnet_mel_loss': postnet_mel_loss.item(),
            'stop_loss': stop_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict

def load_model(config: Dict, checkpoint_path: Optional[str] = None) -> StutterTTS:
    """Helper function to load model and optionally checkpoint"""
    model = StutterTTS(config)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
