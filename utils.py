"""
Utility functions for saving and loading model components
"""
import pickle
import torch
from typing import Dict, Any

def save_vocabularies(SRC_vocab, TRG_vocab, path: str):
    """
    Save vocabularies to a pickle file
    
    Args:
        SRC_vocab: Source vocabulary
        TRG_vocab: Target vocabulary
        path: Path to save file
    """
    vocab_data = {
        'SRC_vocab': SRC_vocab,
        'TRG_vocab': TRG_vocab
    }
    with open(path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabularies saved to {path}")

def load_vocabularies(path: str):
    """
    Load vocabularies from a pickle file
    
    Args:
        path: Path to vocabulary file
    
    Returns:
        Tuple of (SRC_vocab, TRG_vocab)
    """
    with open(path, 'rb') as f:
        vocab_data = pickle.load(f)
    return vocab_data['SRC_vocab'], vocab_data['TRG_vocab']

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    SRC_vocab,
    TRG_vocab,
    config,
    path: str
):
    """
    Save a complete checkpoint including model, optimizer, and vocabularies
    
    Args:
        model: Transformer model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        valid_loss: Validation loss
        SRC_vocab: Source vocabulary
        TRG_vocab: Target vocabulary
        config: Model configuration
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'SRC_vocab': SRC_vocab,
        'TRG_vocab': TRG_vocab,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path: str, device: str = 'cpu'):
    """
    Load a complete checkpoint
    
    Args:
        path: Path to checkpoint file
        device: Device to load to
    
    Returns:
        Dictionary containing all checkpoint data
    """
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }

def print_model_summary(model):
    """
    Print a summary of the model architecture
    
    Args:
        model: PyTorch model
    """
    print("\nModel Architecture Summary:")
    print("=" * 60)
    
    total_params = 0
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        total_params += num_params
        print(f"{name:20} | Parameters: {num_params:,}")
    
    print("=" * 60)
    print(f"{'Total':20} | Parameters: {total_params:,}")
    print()
