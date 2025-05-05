import logging
from typing import Any

from fuzzyai.handlers.mutators.base import BaseMutator, mutators_fm
from fuzzyai.handlers.mutators.enums import MutatorType
from fuzzyai.llm.providers.base import BaseLLMProvider

PROMPT_TEMPLATE = "Rephrase the following sentence: {question}"

logger = logging.getLogger(__name__)

@mutators_fm.flavor(MutatorType.REPHRASE)
class RephraseMutator(BaseMutator):
    """
    A mutator that rephrases a given prompt using a language model.

    Args:
        llm (BaseLLMProvider): The language model provider used for rephrasing.
        **extra (Any): Additional arguments to be passed to the BaseMutator constructor.

    Attributes:
        _llm (BaseLLMProvider): The language model provider used for rephrasing.

    """

    def __init__(self, llm: BaseLLMProvider, **extra: Any):
        super().__init__(name=MutatorType.REPHRASE, **extra)
        self._llm = llm
        
    async def mutate(self, prompt: str) -> str:
        """
        Mutates the given prompt by rephrasing it using the language model.

        Args:
            prompt (str): The prompt to be rephrased.

        Returns:
            str: The rephrased prompt.

        """
        logger.debug("Rephrasing prompt: %s", prompt)
        response = await self._llm.generate(PROMPT_TEMPLATE.format(question=prompt), **self._extra)
        logger.debug("Rephrased prompt: %s", response.response if response else prompt)
        return response.response if response else prompt