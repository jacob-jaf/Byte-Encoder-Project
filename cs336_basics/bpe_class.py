from dataclasses import dataclass

@dataclass
class bpe_tokenizer:
    #Reminder: When I use the @dataclass decorator, 
    #   I don't need to explicitly write an __init__ method, 
    #   one is automatically generated for me,
    #   defining self.vocab = vocab and same for merges 
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    @classmethod 
    def trainer(cls, text_data: bytes | str, v_size):
        if isinstance(text_data, str):
            text_data = text_data.encode("utf-8")
        