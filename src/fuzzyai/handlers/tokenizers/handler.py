# type: ignore

from typing import Union

import tiktoken
import torch
from transformers import AutoTokenizer


class TokensHandler:
    def __init__(
        self,
        tokenizer: Union[AutoTokenizer, tiktoken.Encoding],
    ):
        self._tokenizer = tokenizer
        self.vocabulary_size = -1
        self.encode = None
        self.decode = None
        self.batch_decode = None
        
        if isinstance(self._tokenizer, tiktoken.Encoding):
            self.vocabulary_size = self._tokenizer.n_vocab
            self.encode = lambda x: torch.Tensor(self._tokenizer.encode(x)).type(torch.int64)
            self.decode = self._tokenizer.decode
            self.batch_decode = lambda x: self._tokenizer.decode_batch(x.tolist())
        else:
            self.vocabulary_size = self._tokenizer.vocab_size 
            self.encode = lambda x: self._tokenizer.encode(x, return_tensors="pt") 
            self.decode = lambda x, **kwargs: self._tokenizer.decode(x, **kwargs)
            self.batch_decode = self._tokenizer.batch_decode

    def generate_suffixes(
        self,
        seed: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        batch_size: int = 256,
        append_seed: bool = True,
    ) -> tuple[torch.Tensor, list[str]]:
        encoded_seed = self.encode(seed)
        encoded_seed = encoded_seed.squeeze(0)[1:]  # remove the bos token
        # Repeat the encoded seed to create a batch
        batch_size = batch_size if not append_seed else batch_size - 1
        original_tokens_batch = encoded_seed.repeat(batch_size, 1)

        # Calculate new token positions
        new_token_pos = torch.arange(
            0,
            len(encoded_seed),
            len(encoded_seed) / batch_size,
        ).long()  # Ensure it's a long tensor for indices

        # Randomly sample new token values within the range of the vocabulary size
        new_token_val = torch.randint(0, self.vocabulary_size, (batch_size, 1))
        # Scatter the new token values into the original control tokens at the new positions
        new_tokens_batch = original_tokens_batch.scatter(1, new_token_pos.unsqueeze(-1), new_token_val)
        if append_seed:
            new_tokens_batch = torch.cat((encoded_seed.unsqueeze(0), new_tokens_batch), dim=0)
        
        decoded_suffixes: list[str] = self.batch_decode(new_tokens_batch)

        return new_tokens_batch, decoded_suffixes
