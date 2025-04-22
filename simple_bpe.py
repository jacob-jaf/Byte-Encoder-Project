from collections import defaultdict
from typing import Dict, List, Tuple
import re

class SimpleBPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}  # token_id -> token
        self.merges = []  # list of merge operations
        self.token_to_id = {}  # token -> token_id
        self.special_tokens = set()
        
    def train(self, text: str, special_tokens: List[str] = None):
        """Train the BPE tokenizer on the given text."""
        if special_tokens:
            self.special_tokens = set(special_tokens)
            # Add special tokens to vocab first
            for token in special_tokens:
                self._add_token(token)
        
        # Initialize vocabulary with individual characters
        words = self._get_words(text)
        self._initialize_vocab(words)
        
        # Perform BPE training
        while len(self.vocab) < self.vocab_size:
            # Find most frequent pair
            pairs = self._get_stats(words)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Merge the best pair
            words = self._merge_pair(words, best_pair)
            
            # Add new token to vocab
            new_token = best_pair[0] + best_pair[1]
            self._add_token(new_token)
            self.merges.append(best_pair)
    
    def _get_words(self, text: str) -> List[str]:
        """Split text into words, preserving special tokens."""
        # Split on whitespace and punctuation
        words = []
        current_word = []
        
        for char in text:
            if char.isspace() or char in '.,!?;:()[]{}"\'`':
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
                words.append(char)
            else:
                current_word.append(char)
        
        if current_word:
            words.append(''.join(current_word))
            
        return words
    
    def _initialize_vocab(self, words: List[str]):
        """Initialize vocabulary with individual characters."""
        for word in words:
            if word in self.special_tokens:
                self._add_token(word)
            else:
                for char in word:
                    self._add_token(char)
    
    def _add_token(self, token: str):
        """Add a token to the vocabulary if it doesn't exist."""
        if token not in self.token_to_id:
            token_id = len(self.vocab)
            self.vocab[token_id] = token
            self.token_to_id[token] = token_id
    
    def _get_stats(self, words: List[str]) -> Dict[Tuple[str, str], int]:
        """Get frequency of adjacent pairs in the vocabulary."""
        pairs = defaultdict(int)
        for word in words:
            if word in self.special_tokens:
                continue
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs
    
    def _merge_pair(self, words: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge the given pair in all words."""
        new_words = []
        for word in words:
            if word in self.special_tokens:
                new_words.append(word)
                continue
                
            symbols = list(word)
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    symbols[i:i + 2] = [pair[0] + pair[1]]
                else:
                    i += 1
            new_words.append(''.join(symbols))
        return new_words
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        words = self._get_words(text)
        token_ids = []
        
        for word in words:
            if word in self.special_tokens:
                token_ids.append(self.token_to_id[word])
                continue
                
            # Start with individual characters
            symbols = list(word)
            
            # Apply merges in order
            for pair in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                        symbols[i:i + 2] = [pair[0] + pair[1]]
                    else:
                        i += 1
            
            # Convert symbols to token IDs
            for symbol in symbols:
                token_ids.append(self.token_to_id[symbol])
                
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        return ''.join(self.vocab[token_id] for token_id in token_ids)

# Example usage
if __name__ == "__main__":
    # Create tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=100)
    
    # Train on example text
    text = "Hello, how are you? I am doing well. This is a test of the BPE tokenizer."
    tokenizer.train(text, special_tokens=["<|endoftext|>"])
    
    # Encode some text
    encoded = tokenizer.encode("Hello, how are you?")
    print("Encoded:", encoded)
    
    # Decode back to text
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded) 