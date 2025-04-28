import os
from typing import BinaryIO
import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import random

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

## Usage
with open(..., "rb") as f:
    boundaries = find_chunk_boundaries(
        f, num_processes, "<|endoftext|>".encode("utf-8"))
        
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

        
with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    head = [next(f).strip() for _ in range(10)]





PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
example_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"







@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index



def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    """Start with the list of bytes of string."""
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    for i in range(num_merges):
        """Count the number of occurrences of each pair of tokens"""
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts
        """Find the most common pair."""
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair
        """Merge that pair."""
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices
    return BPETokenizerParams(vocab=vocab, merges=merges)

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

@dataclass
class BPETokenizer:
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def merge_attempt(string: str, num_iter: int):
    "Start by converting all strings into the int that represents which of 256 bytes they are"
    indices_for_bytes = list(map(int, string.encode('utf-8')))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    for i in range(num_iter):
        print(f'On merge number: {i}')
        count_current = defaultdict(int)

        for iter1,iter2 in zip(indices_for_bytes, indices_for_bytes[1:]):
            count_current[(iter1, iter2)] += 1
        #this returns the dictionary element with the highest key
        max_count = max(count_current, key = count_current.get)
        print(f'count_current: {count_current}')
        print(f'max count: {max_count}')
        max_first, max_second = max_count
        idx_to_add = 256 + i
        merges[max_count] = idx_to_add
        vocab[idx_to_add] = vocab[max_first] + vocab[max_second]
        #now want to replace instances of max_first, max_second with combined index
        
        indices_for_bytes = merge_insert(indices_for_bytes, max_first, max_second, idx_to_add)
        print(f'Current merges: {merges}')
    return (vocab, merges)
                #check if we've hit the max

def merge_insert(indices: list[int], first_comp: int, second_comp: int, new_idx: int):
    new_indices = []
    j = 0
    while j <= (len(indices) - 1):
        if j+1 < len(indices) and indices[j] == first_comp and indices[j+1] == second_comp:
            new_indices.append(new_idx)
            j+=2
        else:
            new_indices.append(indices[j])
            j+=1
    return new_indices

def encode(string_to_encode: str, merges_previous: dict[tuple[int, int], int]) -> list[int]:
    indices_bytes = list(map(int, string_to_encode.encode('utf-8')))
    for items_merged, new_item in merges_previous.items():
        indices_bytes = merge_insert(indices_bytes, items_merged[0], items_merged[1], new_item)
    return indices_bytes

def decode(vocab_previous: dict[int, bytes], indices: list[int]) -> str:
    bytes_as_list = list(map(vocab_previous.get, indices))
    bytes_regained = b"".join(bytes_as_list).decode('utf-8')
    return bytes_regained



ma1 = merge_attempt(example_text, 3)

params = train_bpe(example_text, num_merges=3)
tokenizer = BPETokenizer(params)
tokenizer.encode(example_text)
tokenizer.decode(tokenizer.encode(example_text))

encode(example_text, ma1[1])
decode(ma1[0], encode(example_text, ma1[1]))


class BPE_example:
    def __init__(self, vocab_size: int = 257):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.special_tokens = set() 
    
    def train(self, text: str, special_tokens: list[str] = None):
        if special_tokens:
            self.special_tokens = set(special_tokens)

            for token in special_tokens:
                self._add_token(token)

        words = self._get_words(text)

    def _get_words(self, text:str) -> List[str]:
        words = text.findall(PAT, text)
        return words
    
    def _add_token(self, token: bytes):

        if token not in self.token_to_id:
            token_to_id

    def _intialize_vocab(self, words: List[str]):
