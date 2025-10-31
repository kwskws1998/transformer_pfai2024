"""
Configuration settings for the Transformer model
"""
import torch
from dataclasses import dataclass

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
    
    # Training settings
    batch_size: int = 32  # Reduced for Mac M-series stability
    n_epochs: int = 100
    learning_rate: float = 5e-4
    gradient_clip: float = 1.0
    
    # Device settings
    device: str = 'mps' if torch.mps.is_available() else 'cpu'
    
    def __post_init__(self):
        # Check for MPS (Apple Silicon) support
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
    
    # Data paths
    data_root: str = './multi30k-datase/data/task1/raw'
    
    # Vocab settings
    min_freq: int = 2
