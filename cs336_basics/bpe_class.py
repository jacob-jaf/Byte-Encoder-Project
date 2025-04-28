from dataclasses import dataclass
import time
import multiprocessing
import math


@dataclass
class bpe_tokenizer:
    #Reminder: When I use the @dataclass decorator, 
    #   I don't need to explicitly write an __init__ method, 
    #   one is automatically generated for me,
    #   defining self.vocab = vocab and same for merges 
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None: 
            special_tokens = []

    @classmethod 
    def trainer(cls, text_data: bytes | str, v_size:int, special_tokens:list[str] | None = none):
        if isinstance(text_data, str):
            text_data = text_data.encode("utf-8")
        if special_tokens is None:
            special_tokens = []
        start_time = time.perf_counter()
        vocab, merges = bpe_trained_python(text_data, v_size, special_tokens)
        end_time = time.perf_counter()
        print(f"Elapsed time training bpe: {end_time - start_time:.6f} seconds")
        return cls(vocab, merges, special_tokens)
    
    def bpd_trained_python(text_data: bytes, v_size:int, special_tokens:list[str]):
        available_threads = multiprocessing.cpu_count()

        size_chunk = math.ceil(len(text_data) // available_threads)



