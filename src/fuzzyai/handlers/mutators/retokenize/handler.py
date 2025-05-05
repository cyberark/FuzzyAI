import logging
from typing import Any

import sentencepiece as spm

from fuzzyai.handlers.mutators.base import BaseMutator, mutators_fm
from fuzzyai.handlers.mutators.enums import MutatorType

logger = logging.getLogger(__name__)

@mutators_fm.flavor(MutatorType.RETOKENIZE)
class RetokenizeMutator(BaseMutator):
    """
    A mutator that retokenizes a given prompt using a pretrained tokenizer model.
    """

    def __init__(self, **extra: Any):
        super().__init__(name=MutatorType.RETOKENIZE, **extra)
        
    async def mutate(self, prompt: str) -> str:
        """
        Mutates the given prompt by retokenizing it using a pretrained tokenizer model.

        Args:
            prompt (str): The prompt to be retokenized.

        Returns:
            str: The retokenized prompt.
        """
        logger.debug("Retoknizing prompt: %s", prompt)
        # Pretrained tokenizer model for Alice in Wonderland
        # https://github.com/maxtlw/sp_tf2_medium/
        s: spm.SentencePieceProcessor = spm.SentencePieceProcessor(model_file='resources/sp_alice.model')
        # Retokenize the prompt using bpe-dropout
        result: list[str] = s.encode(prompt, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
        result = [x.replace('▁', '') for x in result if x != '▁']
        logger.debug("Retokenized prompt: %s", result)
        return " ".join(result)

# Add asyncio main
import asyncio

if __name__ == '__main__':
    mutator = RetokenizeMutator()
    asyncio.run(mutator.mutate('Who was the first president of the united states of america?'))