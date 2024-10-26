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
    
    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor, 
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # (batch, head, time1, d_k) x (batch, head, d_k, time2) -> (batch, head, time1, time2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)  # (batch, head, time1, d_v)
        
        return output

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_model = d_model
        
        self.w_q = nn.Linear(d_model, d_model)#n_head * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_model)#n_head * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_model)#n_head * self.d_v, bias=False)

        self.fc = nn.Linear(d_model,d_model)#n_head * self.d_v, d_model, bias=False)
        
        #self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        #batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        batch_size = q.size(0)
        
        residual = q
        
        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        # Scale factor
        scaling = float(self.d_k) ** -0.5
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.dropout(self.fc(context))
        
        # Add residual and normalize
        return self.layer_norm(output + residual)

        """
        q = self.w_q(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_k(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_v(v).view(batch_size, len_v, n_head, d_v)
        
        # Transpose for attention calculation
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting
        
        # Apply attention
        output = self.attention(q, k, v, mask=mask)
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        return output
        """

class RelativePositionMultiHeadAttention(nn.Module):
    """Relative Position-based Multi-Head Attention"""
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1, max_relative_position: int = 100):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.max_relative_position = max_relative_position
        
        self.w_q = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.w_r = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        
        # Relative position embedding
        self.rel_embeddings = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, self.d_k)
        )
        nn.init.xavier_uniform_(self.rel_embeddings)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Convert relative position to bucket index"""
        relative_bucket = relative_position.clamp(
            -self.max_relative_position,
            self.max_relative_position
        )
        return relative_bucket + self.max_relative_position
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        
        # Linear projections
        q = self.w_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.w_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.w_v(v).view(batch_size, len_v, self.n_head, self.d_v)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_v)
        
        # Prepare relative position embeddings
        position_ids = torch.arange(len_q, device=q.device)[:, None] - \
                      torch.arange(len_k, device=k.device)[None, :]
        relative_position_bucket = self._relative_position_bucket(position_ids)
        relations_keys = self.rel_embeddings[relative_position_bucket]
        
        # Calculate attention scores
        content_score = torch.matmul(q, k.transpose(-2, -1))
        position_score = torch.matmul(q, relations_keys.transpose(-2, -1))
        
        attention_scores = (content_score + position_score) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        output = torch.matmul(attention_probs, v)
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        return output

# Attention関連のユーティリティ関数
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """パディングマスクの作成"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size: int) -> torch.Tensor:
    """先読み防止用マスクの作成"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return torch.zeros_like(mask).masked_fill(mask == 1, float('-inf'))

def create_combined_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """パディングマスクと先読み防止マスクの組み合わせ"""
    pad_mask = create_padding_mask(seq, pad_idx)
    look_ahead_mask = create_look_ahead_mask(seq.size(1))
    return pad_mask & look_ahead_mask