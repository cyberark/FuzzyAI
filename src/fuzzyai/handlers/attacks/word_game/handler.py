import logging
from typing import Any, Final, Optional, Type

from pydantic import BaseModel, Field

from fuzzyai.consts import DEFAULT_OPEN_SOURCE_MODEL
from fuzzyai.enums import LLMRole
from fuzzyai.handlers.attacks.base import (BaseAttackTechniqueHandler, BaseAttackTechniqueHandlerException,
                                           attack_handler_fm)
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.attacks.word_game.word_game_prompts import (HINTS_GENERATION_PROMPT, REPHRASING_PROMPT,
                                                                  WORD_GAME_PLUS_PROMPT, WORD_GAME_PROMPT,
                                                                  WORD_IDENTIFICATION_PROMPT)
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider

logger = logging.getLogger(__name__)

DEFAULT_GAME_GENERATION_MODEL: Final[str] = DEFAULT_OPEN_SOURCE_MODEL

class WordGameException(BaseAttackTechniqueHandlerException):
    pass

class WordGameAttackHandlerExtraParams(BaseModel):
    word_game_plus: bool = Field(True, description="Use the word game plus variation instead of the regular word game attack. Default: True")
    game_generation_model: str = Field(DEFAULT_GAME_GENERATION_MODEL, description=f"Extract the needed titles from the original prompt. Default: {DEFAULT_GAME_GENERATION_MODEL}")


@attack_handler_fm.flavor(FuzzerAttackMode.WORDGAME)
class WordGameAttackHandler(BaseAttackTechniqueHandler[WordGameAttackHandlerExtraParams]):
    """
    Word Game attack handler - bypass the safety measures of LLMs by simultaneously obfuscating both the input queries and the expected responses.
    """
    def __init__(self, **extra: Any):
        super().__init__(**extra)
        self._word_identification_prompt_response: Optional[BaseLLMProviderResponse] = None
        self._rephrasing_prompt_response: Optional[BaseLLMProviderResponse] = None
        self._hints_generation_prompt_response: Optional[BaseLLMProviderResponse] = None
        if (model := self._extra_args.game_generation_model) not in self._model_queue_map:
            raise RuntimeError(f"game_generation_model: {model} was not added to the fuzzer,"
                               " please make sure you add it with -x <provider/model> and set"
                               " -e game_generation_model=<provider/model> accordingly"
                               f" (you can omit -e if using the default generation model {DEFAULT_GAME_GENERATION_MODEL}.")

    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        chat_messages: list[BaseLLMMessage] = [BaseLLMMessage(role=LLMRole.USER, content=WORD_IDENTIFICATION_PROMPT.format(query=prompt))]
        llm: BaseLLMProvider

        logger.info("Generating wordgame")
        async with self._borrow(self._extra_args.game_generation_model) as llm:
            self._word_identification_prompt_response = await llm.chat(chat_messages)
            if self._word_identification_prompt_response is None:
                raise WordGameException("Word identification failed got None instead")
            chat_messages.append(BaseLLMMessage(role=LLMRole.ASSISTANT, content=self._word_identification_prompt_response.response))

            chat_messages.append(BaseLLMMessage(role=LLMRole.USER, content=REPHRASING_PROMPT))
            self._rephrasing_prompt_response = await llm.chat(chat_messages)
            if self._rephrasing_prompt_response is None:
                raise WordGameException("Prompt rephrasing failed got None instead")
            chat_messages.append(BaseLLMMessage(role=LLMRole.ASSISTANT, content=self._rephrasing_prompt_response.response))

            chat_messages.append(BaseLLMMessage(role=LLMRole.USER, content=HINTS_GENERATION_PROMPT))
            self._hints_generation_prompt_response = await llm.chat(chat_messages)
            if self._hints_generation_prompt_response is None:
                raise WordGameException("Hints generation failed got None instead")

        logger.info("Running wordgame")
        async with self._borrow(self._model) as llm:
            if self._extra_args.word_game_plus:
                changed_prompt = WORD_GAME_PLUS_PROMPT.format(
                    word_game=WORD_GAME_PROMPT.format(
                        word_game=self._hints_generation_prompt_response.response,
                        masked_query=self._rephrasing_prompt_response.response))
            else:
                changed_prompt = WORD_GAME_PROMPT.format(
                    word_game=self._hints_generation_prompt_response.response,
                    masked_query=self._rephrasing_prompt_response.response)

            response = await llm.generate(changed_prompt, **self._extra)
            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt=changed_prompt,
                                       response=response.response) if response else None
            logger.debug("Response: %s", response.response if response else "None")

        classifications = await self._classify_llm_response(response, original_prompt=prompt)
        
        if result:
            result.classifications = classifications

        return result

    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return WordGameAttackHandlerExtraParams