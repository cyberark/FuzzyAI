import logging
from typing import Optional

from fuzzyai.enums import LLMRole
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderResponse

logger = logging.getLogger(__name__)
IMPROVE_PROMPT = "Be more specific about each step"

class RefinementException(Exception):
    """
    Exception for failed refinement of model response.
    """
    ...


class ResponseRefinementHandler:

    def __init__(self, improve_attempts: int = 3):
        self._improve_attempts = improve_attempts

    async def refine_response(self, model: BaseLLMProvider, original_prompt: str, response: Optional[str] = None) -> list[str]:
        chat_messages: list[BaseLLMMessage] = [BaseLLMMessage(role=LLMRole.USER, content=original_prompt)]
        all_responses: list[str] = []

        if response is None:
            logger.debug(f"Generating a response to: {original_prompt}")
            generated_response: Optional[BaseLLMProviderResponse] = await model.chat(messages=chat_messages)
            if generated_response is None or generated_response.response is None:
                raise RefinementException("Failed generating a response from the model, got: None")
            response = generated_response.response

        chat_messages.append(BaseLLMMessage(role=LLMRole.ASSISTANT, content=response))

        logger.debug("Refining previous response for prompt {original_prompt}")
        for _ in range(self._improve_attempts):
            chat_messages.append(BaseLLMMessage(role=LLMRole.USER, content=IMPROVE_PROMPT))
            improved_response = await model.chat(messages=chat_messages)
            if improved_response is None or improved_response.response is None:
                raise RefinementException("Cannot refine response, got no response from the model")
            all_responses.append(improved_response.response)
            chat_messages.append(BaseLLMMessage(role=LLMRole.ASSISTANT, content=improved_response.response))

        return all_responses