import logging
import random
from typing import Any

from fuzzyai.handlers.mutators.base import BaseMutator, mutators_fm
from fuzzyai.handlers.mutators.enums import MutatorType

logger = logging.getLogger(__name__)

@mutators_fm.flavor(MutatorType.RANDROP)
class RandropMutator(BaseMutator):
    def __init__(self, **extra: Any):
        super().__init__(name=MutatorType.RANDROP, **extra)
        
    async def mutate(self, prompt: str) -> str:
        logger.debug("Randropping prompt: %s", prompt)
        # Remove one random word from the prompt
        words = prompt.split()
        if len(words) == 1:
            return prompt
        
        words.pop(random.randint(0, len(words) - 1))
        logger.debug("Randropped prompt: %s", words)
        return " ".join(words)