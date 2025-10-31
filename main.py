"""
Main script to train the Transformer model on Multi30k dataset - MPS Optimized
"""
import os
import torch
import argparse
from pathlib import Path

# Set MPS environment variable BEFORE any imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from config import TransformerConfig
from data_utils import create_data_iterators
from model import Transformer
from train_utils import train_model, test_model
from visualization import (
    visualize_positional_embeddings, 
    plot_training_curves,
    visualize_translation_attention
)

def main(args):
    """Main training function - MPS optimized"""
    
    # Create config
    config = TransformerConfig()
    
    # Force MPS as requested
    if torch.backends.mps.is_available():
        config.device = 'mps'
        print("\n✅ Using MPS (Metal Performance Shaders) backend")
        print("   If you encounter errors, the environment variable")
        print("   PYTORCH_ENABLE_MPS_FALLBACK=1 has been set")
    else:
        print("\n⚠️  MPS not available, using CPU")
        config.device = 'cpu'
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.n_epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.data_root:
        config.data_root = args.data_root
    
    print("\nConfiguration:")
    print("-" * 50)
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    # Check if data exists
    if not os.path.exists(config.data_root):
        print(f"Data not found at {config.data_root}")
        print("Please run: bash prepare_data.sh")
        return
    
    # Create data iterators (now with num_workers=0 in data_utils.py)
    print("\nPreparing data...")
    train_iterator, valid_iterator, test_iterator, SRC_vocab, TRG_vocab = \
        create_data_iterators(config)
    
    print(f"Source vocabulary size: {len(SRC_vocab.itos)}")
    print(f"Target vocabulary size: {len(TRG_vocab.itos)}")
    print(f"Train batches: {len(train_iterator)}")
    print(f"Valid batches: {len(valid_iterator)}")
    print(f"Test batches: {len(test_iterator)}")
    
    # Create model
    print("\nInitializing model...")
    model = Transformer(SRC_vocab, TRG_vocab, config)
    
    # Move model to MPS device
    model = model.to(config.device)
    print(f"Model moved to {config.device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    if not args.test_only:
        print("\nStarting training on MPS...")
        print("Note: If training fails, try reducing batch_size with --batch_size 8")
        save_path = args.save_path if args.save_path else "transformer_best.pt"
        
        try:
            train_losses, valid_losses = train_model(
                model, 
                train_iterator, 
                valid_iterator, 
                config,
                save_path=save_path
            )
            
            # Plot training curves
            if args.plot:
                print("\nPlotting training curves...")
                plot_training_curves(train_losses, valid_losses, 
                                   save_path="training_curves.png")
                
        except RuntimeError as e:
            if "MPS" in str(e) or "mps" in str(e):
                print(f"\n❌ MPS Error encountered: {e}")
                print("\nTry these solutions:")
                print("1. Reduce batch size: --batch_size 8")
                print("2. Restart Python kernel and try again")
                print("3. Update PyTorch: pip install --upgrade torch")
            else:
                raise e
    
    # Load best model if saved
    if args.save_path and os.path.exists(args.save_path):
        print(f"\nLoading best model from {args.save_path}")
        checkpoint = torch.load(args.save_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test model
    print("\nEvaluating on test set...")
    test_loss = test_model(model, test_iterator, config)
    
    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # Visualize positional embeddings
        visualize_positional_embeddings(model, 
                                       save_path="positional_embeddings.png")
        
        # Example translation with attention
        example_sentence = ["ein", "mann", "geht", "die", "straße", "entlang", "."]
        print(f"\nExample translation: {' '.join(example_sentence)}")
        
        trg_tokens, cross_attn, enc_attn = visualize_translation_attention(
            model, 
            example_sentence,
            SRC_vocab,
            TRG_vocab,
            config.device
        )
        print(f"Translation: {' '.join(trg_tokens)}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer on Multi30k - MPS Optimized")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, 
                       help="Path to Multi30k data")
    
    # Model arguments
    parser.add_argument("--save_path", type=str, default="transformer_best.pt",
                       help="Path to save best model")
    parser.add_argument("--test_only", action="store_true",
                       help="Only run testing (load existing model)")
    
    # Visualization arguments
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    parser.add_argument("--plot", action="store_true",
                       help="Plot training curves")
    
    args = parser.parse_args()
    main(args)
