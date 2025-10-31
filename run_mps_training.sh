#!/bin/bash

# MPS-Optimized Transformer Training Setup Script
echo "================================================"
echo "MPS-OPTIMIZED TRANSFORMER TRAINING SETUP"
echo "================================================"

# Set the MPS fallback environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "✅ Set PYTORCH_ENABLE_MPS_FALLBACK=1"

# Check if data exists
if [ ! -d "./multi30k-datase" ]; then
    echo "⚠️  Data not found. Running prepare_data.sh..."
    bash prepare_data.sh
else
    echo "✅ Data directory found"
fi

# Print Python and PyTorch info
echo ""
echo "System Information:"
echo "-------------------"
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"

echo ""
echo "================================================"
echo "READY TO TRAIN WITH MPS"
echo "================================================"
echo ""
echo "To start training, run:"
echo "  python main.py"
echo ""
echo "Or with custom settings:"
echo "  python main.py --batch_size 8 --epochs 10"
echo ""
echo "If you still get bus errors, try:"
echo "  python main.py --batch_size 4"
echo ""
echo "================================================"

# Optional: Start training immediately
read -p "Start training now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python main.py --batch_size 16 --epochs 10
fi
