"""
Encoder and Decoder layers for Transformer model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from modules import MultiHeadAttention, PositionWiseFeedForward, SinusoidalPositionalEmbedding

class EncoderLayer(nn.Module):
    """Single encoder layer"""
    
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.ffn_dim = config.ffn_dim
        
        self.self_attn = MultiHeadAttention(
            emb_dim=self.emb_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.emb_dim)
        self.dropout = config.dropout
        self.activation_fn = nn.ReLU()
        self.PositionWiseFeedForward = PositionWiseFeedForward(
            self.emb_dim, 
            self.ffn_dim, 
            config.dropout
        )
        self.final_layer_norm = nn.LayerNorm(self.emb_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_padding_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of encoder layer"""
        
        residual = x
        x, attn_weights = self.self_attn(
            query=x, 
            key=x, 
            attention_mask=encoder_padding_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        x = self.PositionWiseFeedForward(x)
        x = self.final_layer_norm(x)
        
        # Prevent numerical instability
        if torch.isinf(x).any() or torch.isnan(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        
        return x, attn_weights

class DecoderLayer(nn.Module):
    """Single decoder layer"""
    
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.ffn_dim = config.ffn_dim
        
        self.self_attn = MultiHeadAttention(
            emb_dim=self.emb_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            causal=True,
        )
        self.dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.emb_dim)
        
        self.encoder_attn = MultiHeadAttention(
            emb_dim=self.emb_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.emb_dim)
        
        self.PositionWiseFeedForward = PositionWiseFeedForward(
            self.emb_dim, 
            self.ffn_dim, 
            config.dropout
        )
        self.final_layer_norm = nn.LayerNorm(self.emb_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of decoder layer"""
        
        residual = x
        # Self Attention
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            attention_mask=causal_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        
        # Cross-Attention Block
        residual = x
        x, cross_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)
        
        # Feed Forward
        x = self.PositionWiseFeedForward(x)
        x = self.final_layer_norm(x)
        
        return x, self_attn_weights, cross_attn_weights

class Encoder(nn.Module):
    """Complete encoder stack"""
    
    def __init__(self, config, embed_tokens):
        super().__init__()
        
        self.dropout = config.dropout
        emb_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings
        
        self.embed_tokens = embed_tokens
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings, 
            config.emb_dim, 
            self.padding_idx
        )
        
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.encoder_layers)
        ])
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass of encoder"""
        
        inputs_embeds = self.embed_tokens(input_ids)
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        self_attn_scores = []
        for encoder_layer in self.layers:
            x, attn = encoder_layer(x, attention_mask)
            self_attn_scores.append(attn.detach())
        
        return x, self_attn_scores

class Decoder(nn.Module):
    """Complete decoder stack"""
    
    def __init__(self, config, embed_tokens: nn.Embedding):
        super().__init__()
        
        self.dropout = config.dropout
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        
        self.embed_tokens = embed_tokens
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings, 
            config.emb_dim, 
            self.padding_idx
        )
        
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.decoder_layers)
        ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        decoder_causal_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass of decoder"""
        
        # Embed positions
        positions = self.embed_positions(input_ids)
        x = self.embed_tokens(input_ids)
        x += positions
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Decoder layers
        cross_attention_scores = []
        for idx, decoder_layer in enumerate(self.layers):
            x, layer_self_attn, layer_cross_attn = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                causal_mask=decoder_causal_mask,
            )
            cross_attention_scores.append(layer_cross_attn.detach())
        
        return x, cross_attention_scores
