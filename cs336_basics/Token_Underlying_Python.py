import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path
import multiprocessing
from itertools import chain
import regex as re2

@dataclass
class Token_Underlying:
    """
    A Python implementation of the Rust BPE tokenizer.
    This class provides similar functionality to the Rust implementation,
    though with some Python-specific optimizations.
    """
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]
    special_tokens: List[str]
    
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Sort special tokens by length in descending order
        self.special_tokens.sort(key=lambda x: -len(x))
        
        # Create regex for special tokens
        self.special_regex = None
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.special_regex = re2.compile(pattern)
        
        # Create quick lookup for single-byte tokens
        self.vocab_inv_bytes = [None] * 256
        for token_id, bytes_seq in self.vocab.items():
            if len(bytes_seq) == 1:
                self.vocab_inv_bytes[bytes_seq[0]] = token_id
        
        # Create merges lookup
        self.merges_lookup = {}
        for e1, e2 in self.merges:
            merged = e1 + e2
            e1_token = next(k for k, v in self.vocab.items() if v == e1)
            e2_token = next(k for k, v in self.vocab.items() if v == e2)
            merged_token = next(k for k, v in self.vocab.items() if v == merged)
            self.merges_lookup[(e1_token, e2_token)] = merged_token
        
        # Create inverse vocabulary
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        
        # Create special tokens inverse mapping
        self.special_tokens_inv = {
            token.encode(): self.vocab_inv[token.encode()]
            for token in self.special_tokens
        }
        
        # Main tokenization regex
        self.re = re2.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
    
    def encode(self, text: Union[str, bytes]) -> np.ndarray:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text as string or bytes
            
        Returns:
            numpy array of token IDs
        """
        if isinstance(text, str):
            text = text.encode('utf-8')
            
        if not text:
            return np.array([], dtype=np.uint16)
            
        # Determine number of threads based on text length
        n_threads = multiprocessing.cpu_count() if len(text) > 100_000 else 1
        chunk_size = (len(text) + n_threads - 1) // n_threads
        
        # Find chunk boundaries
        boundaries = [0]
        for i in range(1, n_threads):
            loc = i * chunk_size
            while loc < len(text) and text[loc:loc+1] == b'':
                loc += 1
                
            # Find good chunk boundary
            if self.special_regex:
                match = self.special_regex.search(text[loc:].decode('utf-8', errors='ignore'))
                if match:
                    loc += match.end()
            else:
                # Try to find sentence boundary
                sentence_end = text[loc:].find(b'.\n')
                if sentence_end != -1:
                    loc += sentence_end + 1
                    
            boundaries.append(loc)
        boundaries.append(len(text))
        
        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))
        
        # Create chunk ranges
        chunk_ranges = [(boundaries[i], boundaries[i+1]) 
                       for i in range(len(boundaries)-1)]
        
        # Process chunks in parallel
        with multiprocessing.Pool(n_threads) as pool:
            chunk_results = pool.map(self._process_chunk, 
                                   [(text[start:end], self) for start, end in chunk_ranges])
        
        # Combine results
        tokens = list(chain.from_iterable(chunk_results))
        return np.array(tokens, dtype=np.uint16)
    
    def _process_chunk(self, args: Tuple[bytes, 'RustTokenizer']) -> List[int]:
        """
        Process a single chunk of text.
        """
        chunk, tokenizer = args
        tokens = []
        offset = 0
        
        if tokenizer.special_regex:
            for match in tokenizer.special_regex.finditer(chunk.decode('utf-8', errors='ignore')):
                # Process text before special token
                if match.start() > offset:
                    text = chunk[offset:match.start()]
                    tokens.extend(tokenizer._encode_text(text))
                
                # Add special token
                special_token = tokenizer.special_tokens_inv.get(
                    match.group().encode(),
                    tokenizer.vocab_inv[match.group().encode()]
                )
                tokens.append(special_token)
                offset = match.end()
        
        # Process remaining text
        if offset < len(chunk):
            tokens.extend(tokenizer._encode_text(chunk[offset:]))
            
        return tokens
    
    def _encode_text(self, text: bytes) -> List[int]:
        """
        Encode a piece of text using BPE.
        """
        tokens = []
        for match in self.re.finditer(text.decode('utf-8', errors='ignore')):
            word = match.group().encode()
            if word in self.vocab_inv:
                tokens.append(self.vocab_inv[word])
            else:
                # Apply BPE merges
                current_tokens = [self.vocab_inv_bytes[b] for b in word]
                while len(current_tokens) > 1:
                    # Find best merge
                    best_merge = None
                    best_merge_idx = None
                    
                    for i in range(len(current_tokens) - 1):
                        pair = (current_tokens[i], current_tokens[i + 1])
                        if pair in self.merges_lookup:
                            if best_merge is None or self.merges_lookup[pair] < best_merge:
                                best_merge = self.merges_lookup[pair]
                                best_merge_idx = i
                    
                    if best_merge is None:
                        break
                        
                    # Apply merge
                    current_tokens[best_merge_idx] = best_merge
                    current_tokens.pop(best_merge_idx + 1)
                
                tokens.extend(current_tokens)
                
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text as string
        """
        bytes_seq = b''.join(self.vocab[token] for token in tokens)
        return bytes_seq.decode('utf-8', errors='ignore')
    
    @classmethod
    def from_files(cls, vocab_path: Path, merges_path: Path, 
                   special_tokens: Optional[List[str]] = None) -> 'RustTokenizer':
        """
        Create a tokenizer from saved vocabulary and merges files.
        
        Args:
            vocab_path: Path to vocabulary file
            merges_path: Path to merges file
            special_tokens: Optional list of special tokens
            
        Returns:
            Initialized RustTokenizer instance
        """
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens) 