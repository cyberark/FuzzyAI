import logging
from typing import Any

from fuzzyai.handlers.mutators.base import BaseMutator, mutators_fm
from fuzzyai.handlers.mutators.enums import MutatorType
from fuzzyai.llm.providers.base import BaseLLMProvider

PROMPT_TEMPLATE = "Summarize the following document: {document}"
SYSTEM_PROMPT = "You are proficient in summarizing documents"

PROMPT_TEMPLATE = """I received a prompt.
Can you summarize the core question it asks? The
prompt is below:
{document}
The core question is:"""


logger = logging.getLogger(__name__)

@mutators_fm.flavor(MutatorType.SUMMARIZE)
class SummarizeMutator(BaseMutator):
    """
    A mutator that summarizes a given prompt using a language model.

    Args:
        llm (BaseLLMProvider): The language model provider used for generating summaries.
        **extra (Any): Additional arguments to be passed to the BaseMutator constructor.

    Attributes:
        _llm (BaseLLMProvider): The language model provider used for generating summaries.

    """

    def __init__(self, llm: BaseLLMProvider, **extra: Any):
        super().__init__(name=MutatorType.SUMMARIZE, **extra)
        self._llm = llm
        
    async def mutate(self, prompt: str) -> str:
        """
        Mutates the given prompt by generating a summary using the language model.

        Args:
            prompt (str): The input prompt to be summarized.

        Returns:
            str: The generated summary.

        """
        logger.debug("Summarizing prompt: %s", prompt)
        response = await self._llm.generate(prompt=PROMPT_TEMPLATE.format(document=prompt), **self._extra)
        logger.debug("Summarized prompt: %s", response.response if response else prompt)
        resp = response.response if response else str()
        if resp.startswith("\""):
            resp = resp[1:]
        if resp.endswith("\""):
            resp = resp[:-1]

        return resp