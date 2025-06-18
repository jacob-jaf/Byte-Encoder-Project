from dataclasses import dataclass
import time
import multiprocessing
import math
from pathlib import Path
from collections import Counter, defaultdict
from cs336_basics.Token_Underlying_Python import Tokenizer_Underlying
from cs336_basics.train_bpe_low import train_bpe
import pickle
from typing import Literal, Iterable, Iterator
import numpy as np
import regex as re




@dataclass
class Tokenizer:
    #Reminder: When I use the @dataclass decorator, 
    #   I don't need to explicitly write an __init__ method, 
    #   one is automatically generated for me,
    #   defining self.vocab = vocab and same for merges 

    '''
    Essentially a way to initialize BPE class from different possible starting points
    '''

    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    token_underlined = Tokenizer_Underlying | None

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None: 
            special_tokens = []
        self.token_underlined = Tokenizer_Underlying(vocab, merges, special_tokens)

    @classmethod 
    def trainer(cls, text_data: bytes | str, v_size:int, special_tokens:list[str] | None = None):
        """
        This is for if we receive the text directly
        """
        if special_tokens is None:
            special_tokens = []
        start_time = time.perf_counter()
        print(type(text_data))
        vocab, merges = train_bpe(text_data, v_size, special_tokens)
        end_time = time.perf_counter()
        print(f"Elapsed time training bpe: {end_time - start_time:.6f} seconds")
        return cls(vocab, merges, special_tokens)
    
    @classmethod
    def from_files(cls, file_path_vocab: Path, file_path_merges: Path, special_token: list[str] | None = None):
        """
        This alternate constructor is for when we use file paths for already trained vocab/merges
        """
        with open(file_path_vocab, 'rb') as f:
            vocab = pickle.load(f)
        with open(file_path_merges, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_token)
    
    @classmethod
    def train_from_data_file(cls, load_path: Path | str, vocab_size: int, special_tokens: list[str] | None = None, bytes_to_read = -1):
        with open(load_path, 'rb') as f:
            text_4_train = f.read(bytes_to_read)
            # Ensure we don't cut off in the middle of a UTF-8 character
            try:
                text_4_train.decode('utf-8')
            except UnicodeDecodeError:
                # If we hit a decode error, read until the last complete character
                while True:
                    try:
                        text_4_train.decode('utf-8')
                        break
                    except UnicodeDecodeError:
                        text_4_train = text_4_train[:-1]
        return cls.trainer(text_4_train, vocab_size, special_tokens)
    
    def encode(self, text: str | bytes) -> list[int]:
        if isinstance(text, str):
            text = text.encode('utf-8')
        return self.token_underlined.encode(text)
    
    def decode(self, tokens_list: list[int]) -> str:
        return self.token_underlined.decode(tokens_list)
    
    def encode_iterable(self, iterable: Iterable[str], as_list=True) -> Iterator[int] | Iterator[np.ndarray]:
        for text in iterable:
            yield from self.encode(text, as_list=as_list)
    

    
        

    

    


#python implementations to add: bpe_trained_python

if __name__ == '__main__':
    #with open('data/owt_valid.txt', 'rb') as f:
    #    text_4_train = f.read(2000)

    owt_valid = Tokenizer.train_from_data_file(load_path='data/owt_valid.txt',
                                   vocab_size=32000,
                                   special_tokens=['<|endoftext|>'], 
                                   bytes_to_read=2000)

