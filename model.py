"""
Main Transformer model combining encoder and decoder
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from layers import Encoder, Decoder

class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks"""
    
    def __init__(self, SRC_vocab, TRG_vocab, config):
        super().__init__()
        
        self.SRC_vocab = SRC_vocab
        self.TRG_vocab = TRG_vocab
        self.device = config.device
        
        # Embeddings
        self.enc_embedding = nn.Embedding(
            len(SRC_vocab.itos), 
            config.emb_dim, 
            padding_idx=SRC_vocab.stoi['<pad>']
        )
        self.dec_embedding = nn.Embedding(
            len(TRG_vocab.itos), 
            config.emb_dim, 
            padding_idx=TRG_vocab.stoi['<pad>']
        )
        
        # Encoder and Decoder
        self.encoder = Encoder(config, self.enc_embedding)
        self.decoder = Decoder(config, self.dec_embedding)
        
        # Output projection
        self.prediction_head = nn.Linear(config.emb_dim, len(TRG_vocab.itos))
        
        # Initialize weights
        self.init_weights()
    
    def generate_mask(
        self, 
        src: torch.Tensor, 
        trg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate attention masks for encoder and decoder"""
        
        # Mask encoder attention to ignore padding
        enc_attention_mask = src.eq(self.SRC_vocab.stoi['<pad>']).to(self.device)
        
        # Mask decoder attention for causality
        tmp = torch.ones(trg.size(1), trg.size(1), dtype=torch.bool, device=self.device)
        mask = torch.arange(tmp.size(-1), device=self.device)
        dec_attention_mask = tmp.masked_fill_(
            mask < (mask + 1).view(tmp.size(-1), 1), 
            False
        ).to(self.device)
        
        return enc_attention_mask, dec_attention_mask
    
    def init_weights(self):
        """Initialize model parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)
    
    def forward(
        self, 
        src: torch.Tensor, 
        trg: torch.Tensor
    ) -> Tuple[torch.Tensor, list, list]:
        """Forward pass of Transformer"""
        
        # Generate masks
        enc_attention_mask, dec_causal_mask = self.generate_mask(src, trg)
        
        # Encode
        encoder_output, encoder_attention_scores = self.encoder(
            input_ids=src,
            attention_mask=enc_attention_mask
        )
        
        # Decode
        decoder_output, decoder_attention_scores = self.decoder(
            trg,
            encoder_output,
            encoder_attention_mask=enc_attention_mask,
            decoder_causal_mask=dec_causal_mask,
        )
        
        # Project to vocabulary
        decoder_output = self.prediction_head(decoder_output)
        
        return decoder_output, encoder_attention_scores, decoder_attention_scores
    
    def translate(
        self, 
        src: torch.Tensor, 
        max_length: int = 50, 
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Translate source sequence to target sequence using greedy decoding
        
        Args:
            src: Source sequence tensor
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (1.0 = no change)
        
        Returns:
            Generated target sequence
        """
        self.eval()
        
        with torch.no_grad():
            # Encode source
            enc_attention_mask = src.eq(self.SRC_vocab.stoi['<pad>']).to(self.device)
            encoder_output, _ = self.encoder(
                input_ids=src,
                attention_mask=enc_attention_mask
            )
            
            # Start with SOS token
            trg = torch.LongTensor([[self.TRG_vocab.stoi['<sos>']]]). to(self.device)
            
            for _ in range(max_length):
                # Generate causal mask
                tmp = torch.ones(trg.size(1), trg.size(1), dtype=torch.bool, device=self.device)
                mask = torch.arange(tmp.size(-1), device=self.device)
                dec_causal_mask = tmp.masked_fill_(
                    mask < (mask + 1).view(tmp.size(-1), 1), 
                    False
                )
                
                # Decode
                decoder_output, _ = self.decoder(
                    trg,
                    encoder_output,
                    encoder_attention_mask=enc_attention_mask,
                    decoder_causal_mask=dec_causal_mask,
                )
                
                # Get next token prediction
                logits = self.prediction_head(decoder_output[:, -1:, :])
                logits = logits / temperature
                next_token = torch.argmax(logits, dim=-1)
                
                # Append to target sequence
                trg = torch.cat([trg, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == self.TRG_vocab.stoi['<eos>']:
                    break
            
            return trg
