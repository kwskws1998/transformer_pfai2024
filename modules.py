"""
Core modules for Transformer model: Multi-Head Attention and Position-wise Feed Forward
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        encoder_decoder_attention: bool = False,
        causal: bool = False
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = emb_dim // num_heads
        
        assert self.head_dim * num_heads == self.emb_dim, \
            "emb_dim must be divisible by num_heads"
        
        self.encoder_decoder_attention = encoder_decoder_attention
        self.causal = causal
        
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-headed attention computation"""
        # (batch_size, seq_len, emb_dim) -> (batch_size, num_heads, seq_len, head_dim)
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def MultiHead_scaled_dot_product(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention"""
        
        # QK^T / sqrt(d)
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            if self.causal:
                # For causal mask (seq_len x seq_len)
                attn_weights = attn_weights.masked_fill(
                    attention_mask.unsqueeze(0).unsqueeze(1), 
                    float("-inf")
                )
            else:
                # For encoder mask (batch_size x seq_len)
                attn_weights = attn_weights.masked_fill(
                    attention_mask.unsqueeze(1).unsqueeze(2), 
                    float("-inf")
                )
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, value)
        
        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        concat_attn_output_shape = attn_output.size()[:-2] + (self.emb_dim,)
        attn_output = attn_output.view(*concat_attn_output_shape)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention"""
        
        q = self.q_proj(query)
        
        # Encoder-Decoder attention
        if self.encoder_decoder_attention:
            k = self.k_proj(key)
            v = self.v_proj(key)
        # Self attention
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
        
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        
        attn_output, attn_weights = self.MultiHead_scaled_dot_product(q, k, v, attention_mask)
        return attn_output, attn_weights

class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, emb_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.activation = nn.ReLU()
        self.w_1 = nn.Linear(emb_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, emb_dim)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x
        x = self.activation(self.w_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.w_2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual  # Residual connection

class SinusoidalPositionalEmbedding(nn.Embedding):
    """Fixed sinusoidal positional embeddings"""
    
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)
    
    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """Initialize with sinusoidal patterns"""
        n_pos, embed_dim = out.shape
        pe = nn.Parameter(torch.zeros(out.shape))
        
        for pos in range(n_pos):
            for i in range(0, embed_dim, 2):
                pe[pos, i].data.copy_(
                    torch.tensor(np.sin(pos / (10000 ** (i / embed_dim))))
                )
                if i + 1 < embed_dim:
                    pe[pos, i + 1].data.copy_(
                        torch.tensor(np.cos(pos / (10000 ** ((i + 1) / embed_dim))))
                    )
        
        pe.detach_()
        return pe
    
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate positional embeddings for input"""
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)
