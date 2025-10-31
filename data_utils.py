"""
Data utilities for loading and preprocessing Multi30k dataset
"""
import os
import io
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import Dict, List, Tuple

# Load spaCy models for tokenization
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except OSError:
    print("Please install spaCy models:")
    print("python -m spacy download de_core_news_sm")
    print("python -m spacy download en_core_web_sm")
    raise

def tokenize_de(text: str) -> List[str]:
    """Tokenize German text"""
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

def tokenize_en(text: str) -> List[str]:
    """Tokenize English text"""
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

class TranslationDataset(Dataset):
    """Custom Dataset class for Multi30k translation data"""
    
    def __init__(self, root_dir: str, split: str):
        self.root_dir = root_dir
        self.split = split
        
        self.data_files = {
            'train': ('train.de', 'train.en'),
            'valid': ('val.de', 'val.en'),
            'test': ('test_2016_flickr.de', 'test_2016_flickr.en')
        }
        
        self.de_file_path = os.path.join(self.root_dir, self.data_files[self.split][0])
        self.en_file_path = os.path.join(self.root_dir, self.data_files[self.split][1])
        
        with io.open(self.de_file_path, mode='r', encoding='utf-8') as de_file, \
             io.open(self.en_file_path, mode='r', encoding='utf-8') as en_file:
            self.de_sentences = de_file.readlines()
            self.en_sentences = en_file.readlines()
    
    def __len__(self):
        return len(self.de_sentences)
    
    def __getitem__(self, idx):
        de_sentence = tokenize_de(self.de_sentences[idx].strip())
        en_sentence = tokenize_en(self.en_sentences[idx].strip())
        return {'SRC': de_sentence, 'TRG': en_sentence}

class Vocab:
    """Vocabulary class for managing word-to-index mappings"""
    
    def __init__(self, counter: Counter, min_freq: int):
        self.itos = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.min_freq = min_freq
        self.build_vocab(counter)
    
    def build_vocab(self, counter: Counter):
        """Build vocabulary from word counter"""
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
    
    def numericalize(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]

def build_counter(dataset: Dataset) -> Counter:
    """Build word frequency counter from dataset"""
    counter = Counter()
    for i in range(len(dataset)):
        example = dataset[i]
        counter.update(example['SRC'])
        counter.update(example['TRG'])
    return counter

def create_data_iterators(config) -> Tuple[DataLoader, DataLoader, DataLoader, Vocab, Vocab]:
    """Create data loaders and vocabularies"""
    
    # Create datasets
    train_data = TranslationDataset(root_dir=config.data_root, split='train')
    valid_data = TranslationDataset(root_dir=config.data_root, split='valid')
    test_data = TranslationDataset(root_dir=config.data_root, split='test')
    
    # Build vocabularies
    counter = build_counter(train_data)
    SRC_vocab = Vocab(counter, min_freq=config.min_freq)
    TRG_vocab = Vocab(counter, min_freq=config.min_freq)
    
    # Define collate function
    def collate_fn(batch):
        src_batch = [torch.tensor(SRC_vocab.numericalize(item['SRC'])) for item in batch]
        trg_batch = [torch.tensor(TRG_vocab.numericalize(item['TRG'])) for item in batch]
        
        src_batch_padded = pad_sequence(src_batch, 
                                        padding_value=SRC_vocab.stoi['<pad>'], 
                                        batch_first=True)
        trg_batch_padded = pad_sequence(trg_batch, 
                                        padding_value=TRG_vocab.stoi['<pad>'], 
                                        batch_first=True)
        
        return {'SRC': src_batch_padded, 'TRG': trg_batch_padded}
    
    # Create data loaders
    # Set num_workers=0 for Mac to avoid multiprocessing issues
    num_workers = 0  # Mac M-series compatibility
    
    train_iterator = DataLoader(train_data, 
                                batch_size=config.batch_size, 
                                shuffle=True, 
                                collate_fn=collate_fn,
                                num_workers=num_workers)
    valid_iterator = DataLoader(valid_data, 
                               batch_size=config.batch_size, 
                               shuffle=True, 
                               collate_fn=collate_fn,
                               num_workers=num_workers)
    test_iterator = DataLoader(test_data, 
                              batch_size=config.batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn,
                              num_workers=num_workers)
    
    return train_iterator, valid_iterator, test_iterator, SRC_vocab, TRG_vocab
