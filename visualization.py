"""
Visualization utilities for Transformer model
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Dict, Optional

def visualize_positional_embeddings(model, save_path: Optional[str] = None):
    """
    Visualize the positional embedding matrix
    
    Args:
        model: Transformer model
        save_path: Path to save the figure (optional)
    """
    positional_embeddings = model.encoder.embed_positions.weight.data.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(15, 9))
    cax = ax.matshow(positional_embeddings, aspect='auto', cmap=plt.cm.YlOrRd)
    fig.colorbar(cax)
    ax.set_title('Positional Embedding Matrix', fontsize=18)
    ax.set_xlabel('Embedding Dimension', fontsize=14)
    ax.set_ylabel('Sequence Position', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def visualize_attention_weights(
    attention_scores: torch.Tensor,
    src_tokens: List[str],
    trg_tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights for a specific layer and head
    
    Args:
        attention_scores: Attention weight tensor
        src_tokens: Source tokens
        trg_tokens: Target tokens  
        layer_idx: Which layer to visualize
        head_idx: Which attention head to visualize
        save_path: Path to save the figure (optional)
    """
    # Get specific attention weights
    if len(attention_scores.shape) == 4:  # (batch, heads, seq_len, seq_len)
        attn_weights = attention_scores[0, head_idx, :, :].cpu().numpy()
    else:
        attn_weights = attention_scores.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.matshow(attn_weights, cmap='Blues')
    
    # Set ticks
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='left')
    ax.set_yticklabels(trg_tokens)
    
    # Add colorbar
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # Labels
    ax.set_xlabel('Source Tokens', fontsize=12)
    ax.set_ylabel('Target Tokens', fontsize=12)
    ax.set_title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def plot_training_curves(
    train_losses: List[float],
    valid_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        valid_losses: List of validation losses
        save_path: Path to save the figure (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, valid_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Perplexity plot
    train_ppl = [np.exp(loss) if loss < 100 else np.inf for loss in train_losses]
    valid_ppl = [np.exp(loss) if loss < 100 else np.inf for loss in valid_losses]
    
    ax2.plot(epochs, train_ppl, 'b-', label='Training Perplexity')
    ax2.plot(epochs, valid_ppl, 'r-', label='Validation Perplexity')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def visualize_translation_attention(
    model,
    src_sentence: List[str],
    src_vocab,
    trg_vocab,
    device: str,
    max_length: int = 50
):
    """
    Visualize attention during translation
    
    Args:
        model: Transformer model
        src_sentence: Source sentence (list of tokens)
        src_vocab: Source vocabulary
        trg_vocab: Target vocabulary
        device: Device to run on
        max_length: Maximum translation length
    
    Returns:
        Translated tokens and attention scores
    """
    model.eval()
    
    # Numericalize source sentence
    src_indices = [src_vocab.stoi['<sos>']] + \
                  [src_vocab.stoi.get(token, src_vocab.stoi['<unk>']) for token in src_sentence] + \
                  [src_vocab.stoi['<eos>']]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode source
        enc_attention_mask = src_tensor.eq(src_vocab.stoi['<pad>']).to(device)
        encoder_output, enc_attention_scores = model.encoder(
            input_ids=src_tensor,
            attention_mask=enc_attention_mask
        )
        
        # Initialize translation with SOS token
        trg_indices = [trg_vocab.stoi['<sos>']]
        attention_scores = []
        
        for _ in range(max_length):
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
            
            # Generate causal mask
            tmp = torch.ones(trg_tensor.size(1), trg_tensor.size(1), 
                           dtype=torch.bool, device=device)
            mask = torch.arange(tmp.size(-1), device=device)
            dec_causal_mask = tmp.masked_fill_(
                mask < (mask + 1).view(tmp.size(-1), 1), False
            )
            
            # Decode
            decoder_output, cross_attention_scores = model.decoder(
                trg_tensor,
                encoder_output,
                encoder_attention_mask=enc_attention_mask,
                decoder_causal_mask=dec_causal_mask,
            )
            
            # Store attention scores
            attention_scores.append(cross_attention_scores)
            
            # Get next token
            logits = model.prediction_head(decoder_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).item()
            trg_indices.append(next_token)
            
            # Stop if EOS token
            if next_token == trg_vocab.stoi['<eos>']:
                break
        
        # Convert indices to tokens
        trg_tokens = [trg_vocab.itos[idx] for idx in trg_indices]
        
        return trg_tokens, attention_scores, enc_attention_scores
