import logging
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class PleaseAttackHandlerExtraParams(BaseModel):
    add_prefix: bool = Field(True, description="Adds 'please' before the prompt. default: True")
    add_suffix: bool = Field(True, description="Adds 'please' after the prompt. default: True")

@attack_handler_fm.flavor(FuzzerAttackMode.PLEASE)
class PleaseAttackHandler(BaseAttackTechniqueHandler[PleaseAttackHandlerExtraParams]):
    """
    Please attack handler - adds 'please' to the beginning and end of the sentence.
    """

    def __init__(self, **extra: Any):
        super().__init__(**extra)

    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        llm: BaseLLMProvider

        async with self._borrow(self._model) as llm:
            changed_prompt: str = prompt
            if not self._extra_args.add_suffix and not self._extra_args.add_prefix:
                logger.warning("add_suffix and add_prefix are both false, nothing will be added to the prompt")
            if self._extra_args.add_prefix:
                changed_prompt = f"Please {prompt}"
            if self._extra_args.add_suffix:
                changed_prompt += ", please"

            response = await llm.generate(changed_prompt, **self._extra)
            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt=changed_prompt,
                                       response=response.response) if response else None
            logger.debug("Response: %s", response.response if response else "None")

        classifications = await self._classify_llm_response(response)

        if result:
            result.classifications = classifications
        return result

    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return PleaseAttackHandlerExtraParams