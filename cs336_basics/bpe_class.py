from dataclasses import dataclass
import time
import multiprocessing
import math
from pathlib import Path
from collections import Counter, defaultdict



@dataclass
class bpe_tokenizer:
    #Reminder: When I use the @dataclass decorator, 
    #   I don't need to explicitly write an __init__ method, 
    #   one is automatically generated for me,
    #   defining self.vocab = vocab and same for merges 

    '''
    Essentially a way to initialize BPE class from different possible starting points
    '''

    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    token_underlined = Token_Underlying | None

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None: 
            special_tokens = []
        self.token_underlined = Token_Underlying(vocab, merges, special_tokens)

    @classmethod 
    def trainer(cls, text_data: bytes | str, v_size:int, special_tokens:list[str] | None = none):
        """
        This is for if we receive the text directly
        """
        if isinstance(text_data, str):
            text_data = text_data.encode("utf-8")
        if special_tokens is None:
            special_tokens = []
        start_time = time.perf_counter()
        vocab, merges = bpe_trained_python(text_data, v_size, special_tokens)
        end_time = time.perf_counter()
        print(f"Elapsed time training bpe: {end_time - start_time:.6f} seconds")
        return cls(vocab, merges, special_tokens)
    
    @classmethod
    def files_trainer(cls, file_path_vocab: Path, file_path_merges: Path, special_token: list[str] | None = None):
        """
        This alternate constructor is for when we use file paths for already trained vocab/merges
        """
        with open(file_path_vocab, 'rb') as f:
            text_vocab = pickle.load(f)
        with open(file_path_merges 'rb') as f:
            text_merges = pickle.load(f)
        return cls(vocab, merges, special_token)
    
    def encode(self, text: str | bytes) -> list[int]:
        if isinstance(text, str):
            text = text.encode('utf-8')
        return self.token_underlined.encode().tolist()
    
    def decode(self, tokens_list: list[int]) -> str:
        return self.token_underlined.decode(tokens_list)
    

    
        

    

    

#python implementations to add: bpe_trained_python
