import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, temperature: float, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # デバッグ用

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q: Query tensor [batch, head, time1, dim]
            k: Key tensor [batch, head, time2, dim]
            v: Value tensor [batch, head, time2, dim]
            mask: Mask tensor [batch, 1, time1, time2] or [batch, head, time1, time2]
        """
        # Calculate attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        # Apply mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        self.attention_weights = attn.detach()  # Save for visualization
        
        # Apply attention to values
        output = torch.matmul(attn, v)
        return output

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with proper tensor shapes"""
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Ensure d_model is divisible by n_head
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_model = d_model
        
        # Linear layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Save attention weights for visualization
        self.attention_weights = None

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            q: Query tensor [batch_size, seq_len_q, d_model]
            k: Key tensor [batch_size, seq_len_k, d_model]
            v: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor [batch_size, 1, seq_len_q, seq_len_k]
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
            attention: Attention weights (optional, for visualization)
        """
        batch_size = q.size(0)
        seq_len_q, seq_len_k, seq_len_v = q.size(1), k.size(1), v.size(1)
        
        # Store residual
        residual = q
        
        # Linear projections and split heads
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_head, d_k] -> [batch_size, n_head, seq_len, d_k]
        q = self.w_q(q).view(batch_size, seq_len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len_v, self.n_head, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        # [batch_size, n_head, seq_len_q, d_k] @ [batch_size, n_head, d_k, seq_len_k]
        # -> [batch_size, n_head, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads: [batch_size, 1, 1, seq_len] -> [batch_size, n_head, seq_len_q, seq_len_k]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply attention
        attn = self.dropout(F.softmax(scores, dim=-1))
        self.attention_weights = attn.detach()
        
        # Apply attention to values
        # [batch_size, n_head, seq_len_q, seq_len_k] @ [batch_size, n_head, seq_len_v, d_k]
        # -> [batch_size, n_head, seq_len_q, d_k]
        context = torch.matmul(attn, v)
        
        # Concatenate heads
        # [batch_size, n_head, seq_len_q, d_k] -> [batch_size, seq_len_q, n_head, d_k]
        # -> [batch_size, seq_len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Final linear projection
        output = self.dropout(self.w_o(context))
        
        # Add residual and normalize
        output = self.layer_norm(output + residual)
        
        return output, self.attention_weights

class LocationSensitiveAttention(nn.Module):
    """Location-Sensitive Attention"""
    def __init__(self,
                 attention_dim: int,
                 attention_filters: int = 32,
                 attention_kernel: int = 31,
                 dropout: float = 0.1):
        super().__init__()
        
        self.location_conv = nn.Conv1d(
            in_channels=2,  # Previous attention weights + cumulative weights
            out_channels=attention_filters,
            kernel_size=attention_kernel,
            padding=(attention_kernel - 1) // 2,
            bias=False
        )
        self.location_dense = nn.Linear(attention_filters, attention_dim, bias=False)
        
        self.query_layer = nn.Linear(attention_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(attention_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.score_mask_value = -float("inf")

    def forward(self,
                query: torch.Tensor,
                memory: torch.Tensor,
                attention_weights_cat: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor [batch, 1, attention_dim]
            memory: Memory tensor [batch, max_time, attention_dim]
            attention_weights_cat: Previous and cumulative attention weights [batch, 2, max_time]
            mask: Memory mask [batch, max_time]
        """
        # Process location features
        location_features = self.location_conv(attention_weights_cat)
        location_features = location_features.transpose(1, 2)
        location_features = self.location_dense(location_features)
        
        # Calculate attention scores
        processed_query = self.query_layer(query)
        processed_memory = self.memory_layer(memory)
        
        alignment = processed_memory + processed_query + location_features
        alignment = torch.tanh(alignment)
        alignment = self.v(alignment).squeeze(-1)
        
        # Apply mask
        if mask is not None:
            alignment = alignment.masked_fill(mask == 0, self.score_mask_value)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(alignment, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        
        return attention_context, attention_weights

# Utility functions for creating masks
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Create padding mask"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(seq_len: int) -> torch.Tensor:
    """Create subsequent mask for autoregressive generation"""
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len)), diagonal=1
    ).bool()
    return subsequent_mask

def create_attention_mask(src_seq: torch.Tensor,
                         tgt_seq: Optional[torch.Tensor] = None,
                         pad_idx: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Create masks for encoder-decoder attention"""
    src_mask = create_padding_mask(src_seq, pad_idx)
    
    if tgt_seq is not None:
        tgt_mask = create_padding_mask(tgt_seq, pad_idx)
        subsequent_mask = create_subsequent_mask(tgt_seq.size(1))
        tgt_mask = tgt_mask & subsequent_mask
    else:
        tgt_mask = None
        
    return src_mask, tgt_mask