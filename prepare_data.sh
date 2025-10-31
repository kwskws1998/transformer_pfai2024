#!/bin/bash
# Script to download and prepare Multi30k dataset

echo "Downloading Multi30k dataset..."
git clone --recursive https://github.com/multi30k/dataset.git multi30k-datase

echo "Extracting compressed files..."
find multi30k-datase/ -name '*.gz' -exec gunzip {} \;

echo "Downloading spaCy language models..."
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

echo "Dataset preparation complete!"
