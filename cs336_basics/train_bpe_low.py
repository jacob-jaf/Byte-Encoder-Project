import re
import multiprocessing
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from itertools import chain
import regex as re2
from tqdm import tqdm

@dataclass
class Word:
    symbols: List[int]  # List of token IDs
    word_count: int

def count_words(words: List[str]) -> List[Word]:
    """Count word frequencies and convert to Word objects."""
    word_counts = Counter(words)
    return [
        Word(
            symbols=[ord(c) for c in word],
            word_count=count
        )
        for word, count in word_counts.items()
    ]

def count_pairs(words: List[Word]) -> Dict[Tuple[int, int], int]:
    """Count frequency of adjacent pairs in the vocabulary."""
    pair_counts = defaultdict(int)
    for word in words:
        for i in range(len(word.symbols) - 1):
            pair = (word.symbols[i], word.symbols[i + 1])
            pair_counts[pair] += word.word_count
    return dict(pair_counts)

def update_word(word: Word, pair: Tuple[int, int], new_symbol: int) -> List[Tuple[Tuple[int, int], int]]:
    """Update a word by merging the given pair and return count changes."""
    count_changes = []
    i = 0
    while i < len(word.symbols) - 1:
        if word.symbols[i] == pair[0] and word.symbols[i + 1] == pair[1]:
            # Record changes for adjacent pairs
            if i >= 1:
                count_changes.append(((word.symbols[i - 1], pair[0]), -word.word_count))
                count_changes.append(((word.symbols[i - 1], new_symbol), word.word_count))
            if i <= len(word.symbols) - 3:
                count_changes.append(((pair[1], word.symbols[i + 2]), -word.word_count))
                count_changes.append(((new_symbol, word.symbols[i + 2]), word.word_count))
            
            # Perform the merge
            word.symbols[i] = new_symbol
            word.symbols.pop(i + 1)
        else:
            i += 1
    return count_changes

def update_words(words: List[Word], pair: Tuple[int, int], new_symbol: int) -> Dict[Tuple[int, int], int]:
    """Update all words by merging the given pair and return count changes."""
    count_changes = defaultdict(int)
    
    # Process words in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            update_word,
            [(word, pair, new_symbol) for word in words]
        )
    
    # Aggregate count changes
    for changes in results:
        for (p, change) in changes:
            count_changes[p] += change
    
    return dict(count_changes)

def train_bpe(in_string: str | bytes, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input string.
    
    Args:
        in_string: Input text to train on (string or bytes)
        vocab_size: Desired vocabulary size
        special_tokens: List of special tokens to include in vocabulary
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab is a dict mapping token IDs to bytes
        - merges is a list of tuples (bytes, bytes) representing merge operations
    """
    # Convert bytes to string if needed
    if isinstance(in_string, bytes):
        in_string = in_string.decode('utf-8')
    
    # Initialize regex for tokenization
    re_pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    re_compiled = re2.compile(re_pattern, re2.UNICODE)
    
    # Split text into words
    words = re_compiled.findall(in_string)
    
    # Count words and convert to Word objects
    words = count_words(words)
    print(f"Found {len(words)} unique words")
    
    # Initialize vocabulary with bytes 0-255 and special tokens
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    
    # Initialize merges list
    merges: List[Tuple[bytes, bytes]] = []
    
    # Count initial pairs
    pair_counts = count_pairs(words)
    
    # Training loop
    with tqdm(total=vocab_size - len(vocab), desc="Training BPE") as pbar:
        while len(vocab) < vocab_size and pair_counts:
            # Find most frequent pair
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            
            # Create new token
            new_token = vocab[best_pair[0]] + vocab[best_pair[1]]
            new_token_id = len(vocab)
            vocab[new_token_id] = new_token
            
            # Record merge
            merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
            
            # Update words and get count changes
            count_changes = update_words(words, best_pair, new_token_id)
            
            # Update pair counts
            del pair_counts[best_pair]
            for pair, change in count_changes.items():
                if pair in pair_counts:
                    pair_counts[pair] += change
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                elif change > 0:
                    pair_counts[pair] = change
            
            pbar.update(1)
    
    return vocab, merges 


