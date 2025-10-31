"""
Configuration settings for the Transformer model - MPS optimized version
"""
import torch
from dataclasses import dataclass
import os

# Set environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

@dataclass
class TransformerConfig:
    """Configuration class for Transformer model"""
    # Model dimensions
    emb_dim: int = 64
    ffn_dim: int = 256
    
    # Attention settings
    attention_heads: int = 4
    attention_dropout: float = 0.0
    dropout: float = 0.2
    
    # Position and layer settings
    max_position_embeddings: int = 512
    encoder_layers: int = 3
    decoder_layers: int = 3
    
    # Training settings - Optimized for MPS
    batch_size: int = 16  # Reduced from 32 for MPS stability
    n_epochs: int = 100
    learning_rate: float = 5e-4
    gradient_clip: float = 1.0
    
    # Force MPS device
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Data paths
    data_root: str = './multi30k-datase/data/task1/raw'
    
    # Vocab settings
    min_freq: int = 2
