# transformer_pfai2024

the original code is from https://mp2893.com/course.html
also, in this code repo, we git clone & use the code repo from https://github.com/sjpark9503/attentionviz as well

# Specific detail below:

# Transformer Model for German-to-English Translation

This is a modularized implementation of a Transformer model for machine translation, originally from a Jupyter notebook. The code has been organized into multiple Python files for better maintainability and reusability.

## Project Structure

```
.
├── config.py           # Model and training configuration
├── data_utils.py       # Dataset loading and vocabulary management
├── modules.py          # Core modules (MultiHeadAttention, PositionalEmbedding, etc.)
├── layers.py           # Encoder and Decoder layers
├── model.py            # Main Transformer model
├── train_utils.py      # Training and evaluation functions
├── visualization.py    # Attention and training visualization utilities
├── utils.py            # Helper functions for saving/loading
├── main.py             # Main training script
├── inference.py        # Inference and translation script
├── prepare_data.sh     # Script to download and prepare Multi30k dataset
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

### Python Version
- Python 3.8 or higher is recommended

### Libraries
Install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- torch 2.8.0
- torchaudio 2.8.0
- torchvision 0.23.0
- spaCy 3.5.3
- NumPy 1.24.3
- Matplotlib 3.7.1
- tqdm 4.65.0
- easydict 1.10

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download spaCy language models:**
```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

3. **Download and prepare the Multi30k dataset:**
```bash
bash prepare_data.sh
```

## Training

### Basic training:
```bash
python main.py
```

### Training with custom parameters:
```bash
python main.py --batch_size 64 --epochs 50 --learning_rate 0.0005 --visualize --plot
```

### Available arguments:
- `--batch_size`: Batch size (default: 128)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 5e-4)
- `--data_root`: Path to Multi30k data
- `--save_path`: Path to save best model (default: transformer_best.pt)
- `--test_only`: Only run testing with existing model
- `--visualize`: Generate attention visualizations
- `--plot`: Plot training curves

## Inference

After training, you can use the model for translation:

### Single sentence translation:
```bash
python inference.py --model_path transformer_best.pt --vocab_path vocabularies.pkl \
    --sentence "Ein Mann geht die Straße entlang."
```

### Interactive mode:
```bash
python inference.py --model_path transformer_best.pt --vocab_path vocabularies.pkl \
    --interactive
```

### Translate from file:
```bash
python inference.py --model_path transformer_best.pt --vocab_path vocabularies.pkl \
    --file input.txt --output translations.txt
```

### With beam search:
```bash
python inference.py --model_path transformer_best.pt --vocab_path vocabularies.pkl \
    --sentence "Ein Mann geht die Straße entlang." --beam_search --beam_size 5
```

## Model Architecture

The implementation includes:
- **Multi-Head Self-Attention**: Allows the model to attend to different positions
- **Positional Encoding**: Sinusoidal positional embeddings
- **Encoder-Decoder Architecture**: Standard Transformer architecture
- **Layer Normalization**: Applied after each sub-layer
- **Dropout**: For regularization
- **Residual Connections**: To facilitate training of deep networks

Default configuration:
- Embedding dimension: 64
- Feed-forward dimension: 256
- Attention heads: 4
- Encoder layers: 3
- Decoder layers: 3
- Dropout: 0.2

## Customization

You can modify the model configuration by editing `config.py` or passing arguments to the training script. The modular structure makes it easy to:
- Experiment with different model sizes
- Add new features or modules
- Integrate with other datasets
- Implement different decoding strategies

## Visualization

The code includes utilities for:
- Visualizing positional embeddings
- Plotting training/validation curves
- Displaying attention weights
- Analyzing translation attention patterns

Enable visualizations during training:
```bash
python main.py --visualize --plot
```

## Notes

1. **Memory Requirements**: The default configuration requires approximately 2-4 GB of GPU memory. Reduce batch size if you encounter OOM errors.

2. **Training Time**: On a typical GPU (e.g., RTX 2080), training for 100 epochs takes approximately 1-2 hours.

3. **Data Format**: The Multi30k dataset contains parallel German-English sentences. The code expects the data in the structure created by `prepare_data.sh`.

4. **Saving Vocabularies**: To use the inference script, you need to save vocabularies during training. Modify `main.py` to include:
```python
from utils import save_vocabularies
save_vocabularies(SRC_vocab, TRG_vocab, "vocabularies.pkl")
```

## Troubleshooting

1. **CUDA not available**: The code automatically falls back to CPU if CUDA is not available.

2. **SpaCy models not found**: Make sure to download the language models:
```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

3. **Data not found**: Ensure you've run `prepare_data.sh` and the data is in the correct location.

## License

This implementation is for educational purposes. The Multi30k dataset has its own license terms.
