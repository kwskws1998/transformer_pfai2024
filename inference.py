"""
Inference script for translating sentences with trained Transformer model
"""
import torch
import pickle
import argparse
from typing import List, Optional

from config import TransformerConfig
from model import Transformer
from data_utils import tokenize_de, tokenize_en

class Translator:
    """Class for handling translation with a trained model"""
    
    def __init__(self, model_path: str, vocab_path: Optional[str] = None):
        """
        Initialize translator
        
        Args:
            model_path: Path to saved model checkpoint
            vocab_path: Path to saved vocabularies (optional)
        """
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        
        # Load vocabularies
        if vocab_path:
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                self.SRC_vocab = vocab_data['SRC_vocab']
                self.TRG_vocab = vocab_data['TRG_vocab']
        else:
            # You'll need to save vocabularies during training
            raise ValueError("Vocabulary path required for inference")
        
        # Load model
        self.config = TransformerConfig()
        self.model = Transformer(self.SRC_vocab, self.TRG_vocab, self.config)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def translate_sentence(
        self, 
        sentence: str, 
        max_length: int = 50,
        temperature: float = 1.0,
        beam_size: Optional[int] = None
    ) -> str:
        """
        Translate a sentence
        
        Args:
            sentence: Source sentence in German
            max_length: Maximum length of translation
            temperature: Sampling temperature
            beam_size: Beam search size (None for greedy)
        
        Returns:
            Translated sentence in English
        """
        # Tokenize
        tokens = tokenize_de(sentence)
        
        # Convert to indices
        indices = [self.SRC_vocab.stoi['<sos>']]
        for token in tokens:
            indices.append(self.SRC_vocab.stoi.get(token, self.SRC_vocab.stoi['<unk>']))
        indices.append(self.SRC_vocab.stoi['<eos>'])
        
        # Convert to tensor
        src_tensor = torch.LongTensor(indices).unsqueeze(0).to(self.device)
        
        # Translate
        if beam_size:
            translation = self.beam_search(src_tensor, beam_size, max_length)
        else:
            translation = self.greedy_decode(src_tensor, max_length, temperature)
        
        # Convert back to tokens
        translation_tokens = []
        for idx in translation[0]:
            token = self.TRG_vocab.itos[idx]
            if token == '<eos>':
                break
            if token not in ['<sos>', '<pad>']:
                translation_tokens.append(token)
        
        return ' '.join(translation_tokens)
    
    def greedy_decode(
        self, 
        src: torch.Tensor, 
        max_length: int = 50,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Greedy decoding"""
        with torch.no_grad():
            # Encode
            enc_mask = src.eq(self.SRC_vocab.stoi['<pad>']).to(self.device)
            encoder_output, _ = self.model.encoder(src, enc_mask)
            
            # Start with SOS
            trg = torch.LongTensor([[self.TRG_vocab.stoi['<sos>']]]).to(self.device)
            
            for _ in range(max_length):
                # Create masks
                enc_mask, dec_mask = self.model.generate_mask(src, trg)
                
                # Decode
                decoder_output, _ = self.model.decoder(
                    trg, encoder_output, enc_mask, dec_mask
                )
                
                # Get next token
                logits = self.model.prediction_head(decoder_output[:, -1:, :])
                logits = logits / temperature
                next_token = torch.argmax(logits, dim=-1)
                
                # Append
                trg = torch.cat([trg, next_token], dim=1)
                
                # Stop at EOS
                if next_token.item() == self.TRG_vocab.stoi['<eos>']:
                    break
            
            return trg
    
    def beam_search(
        self, 
        src: torch.Tensor,
        beam_size: int = 5,
        max_length: int = 50
    ) -> torch.Tensor:
        """
        Beam search decoding
        
        Args:
            src: Source tensor
            beam_size: Number of beams
            max_length: Maximum decoding length
        
        Returns:
            Best translation
        """
        with torch.no_grad():
            # Encode
            enc_mask = src.eq(self.SRC_vocab.stoi['<pad>']).to(self.device)
            encoder_output, _ = self.model.encoder(src, enc_mask)
            
            # Expand encoder outputs for beam search
            encoder_output = encoder_output.repeat(beam_size, 1, 1)
            enc_mask = enc_mask.repeat(beam_size, 1)
            
            # Initialize beams
            beams = [(torch.LongTensor([[self.TRG_vocab.stoi['<sos>']]]).to(self.device), 0)]
            complete_beams = []
            
            for step in range(max_length):
                new_beams = []
                
                for beam_seq, beam_score in beams:
                    if beam_seq[0, -1].item() == self.TRG_vocab.stoi['<eos>']:
                        complete_beams.append((beam_seq, beam_score))
                        continue
                    
                    # Decode
                    _, dec_mask = self.model.generate_mask(
                        src[:1].repeat(beam_seq.size(0), 1), beam_seq
                    )
                    decoder_output, _ = self.model.decoder(
                        beam_seq, encoder_output[:beam_seq.size(0)], 
                        enc_mask[:beam_seq.size(0)], dec_mask
                    )
                    
                    # Get probabilities
                    logits = self.model.prediction_head(decoder_output[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Get top k tokens
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        new_seq = torch.cat([
                            beam_seq, 
                            topk_indices[:, i:i+1]
                        ], dim=1)
                        new_score = beam_score + topk_log_probs[0, i].item()
                        new_beams.append((new_seq, new_score))
                
                # Keep top beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_size]
                
                # Early stopping
                if len(complete_beams) >= beam_size:
                    break
            
            # Return best beam
            all_beams = beams + complete_beams
            all_beams.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)  # Normalize by length
            return all_beams[0][0]

def main(args):
    """Main inference function"""
    
    # Initialize translator
    translator = Translator(args.model_path, args.vocab_path)
    
    if args.interactive:
        # Interactive mode
        print("Interactive translation mode (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            sentence = input("\nGerman: ").strip()
            if sentence.lower() == 'quit':
                break
            
            translation = translator.translate_sentence(
                sentence,
                max_length=args.max_length,
                temperature=args.temperature,
                beam_size=args.beam_size if args.beam_search else None
            )
            print(f"English: {translation}")
    
    elif args.file:
        # File translation mode
        with open(args.file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        
        translations = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                translation = translator.translate_sentence(
                    sentence,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    beam_size=args.beam_size if args.beam_search else None
                )
                translations.append(translation)
                print(f"DE: {sentence}")
                print(f"EN: {translation}")
                print()
        
        # Save translations
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(translations))
            print(f"Translations saved to {args.output}")
    
    else:
        # Single sentence translation
        translation = translator.translate_sentence(
            args.sentence,
            max_length=args.max_length,
            temperature=args.temperature,
            beam_size=args.beam_size if args.beam_search else None
        )
        print(f"German: {args.sentence}")
        print(f"English: {translation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate with Transformer model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved model")
    parser.add_argument("--vocab_path", type=str, required=True,
                       help="Path to saved vocabularies")
    
    # Input arguments
    parser.add_argument("--sentence", type=str,
                       help="Single sentence to translate")
    parser.add_argument("--file", type=str,
                       help="File with sentences to translate")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive translation mode")
    
    # Output arguments
    parser.add_argument("--output", type=str,
                       help="Output file for translations")
    
    # Decoding arguments
    parser.add_argument("--max_length", type=int, default=50,
                       help="Maximum translation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--beam_search", action="store_true",
                       help="Use beam search")
    parser.add_argument("--beam_size", type=int, default=5,
                       help="Beam size for beam search")
    
    args = parser.parse_args()
    main(args)
