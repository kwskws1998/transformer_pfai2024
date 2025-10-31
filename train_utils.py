"""
Training and evaluation utilities for Transformer model
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

def train(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: str
) -> float:
    """
    Train the model for one epoch
    
    Args:
        model: Transformer model
        iterator: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        clip: Gradient clipping value
        device: Device to run on
    
    Returns:
        Average epoch loss
    """
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(iterator, desc="Training", leave=False):
        src = batch['SRC'].to(device)
        trg = batch['TRG'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, enc_attention_scores, _ = model(src, trg)
        
        # Reshape for loss calculation (remove last prediction, shift target)
        output = output[:, :-1, :].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(
    model: nn.Module,
    iterator: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Evaluate the model
    
    Args:
        model: Transformer model
        iterator: Validation/test data loader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Average epoch loss
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=False):
            src = batch['SRC'].to(device)
            trg = batch['TRG'].to(device)
            
            # Forward pass
            output, attention_score, _ = model(src, trg)
            
            # Reshape for loss calculation
            output = output[:, :-1, :].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def train_model(
    model: nn.Module,
    train_iterator: DataLoader,
    valid_iterator: DataLoader,
    config,
    save_path: str = None
) -> Tuple[list, list]:
    """
    Complete training loop with early stopping
    
    Args:
        model: Transformer model
        train_iterator: Training data loader
        valid_iterator: Validation data loader
        config: Configuration object
        save_path: Path to save best model
    
    Returns:
        Training and validation loss histories
    """
    device = config.device
    model.to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    PAD_IDX = model.SRC_vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Training loop
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    patience_counter = 0
    patience = 5  # Early stopping patience
    
    print("Starting training...")
    for epoch in range(config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        
        # Train
        train_loss = train(
            model, train_iterator, optimizer, 
            criterion, config.gradient_clip, device
        )
        
        # Validate
        valid_loss = evaluate(model, valid_iterator, criterion, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Calculate perplexity
        train_ppl = math.exp(train_loss) if train_loss < 100 else float('inf')
        valid_ppl = math.exp(valid_loss) if valid_loss < 100 else float('inf')
        
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f}')
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                }, save_path)
                print(f'\tModel saved! Best validation loss: {best_valid_loss:.3f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping after {epoch + 1} epochs')
                break
    
    return train_losses, valid_losses

def test_model(
    model: nn.Module,
    test_iterator: DataLoader,
    config
) -> float:
    """
    Test the model
    
    Args:
        model: Transformer model
        test_iterator: Test data loader
        config: Configuration object
    
    Returns:
        Test loss
    """
    device = config.device
    model.to(device)
    
    PAD_IDX = model.SRC_vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    test_loss = evaluate(model, test_iterator, criterion, device)
    test_ppl = math.exp(test_loss) if test_loss < 100 else float('inf')
    
    print(f'Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f}')
    
    return test_loss
