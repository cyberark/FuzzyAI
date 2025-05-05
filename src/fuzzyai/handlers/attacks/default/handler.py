import logging
from typing import Any, Optional

from pydantic import BaseModel

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

@attack_handler_fm.flavor(FuzzerAttackMode.DEFAULT)
class DefaultAttackHandler(BaseAttackTechniqueHandler[BaseModel]):
    """
    Default attack handler - just generates the prompt as is.
    """
    def __init__(self, **extra: Any):
        super().__init__(**extra)

    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        result: Optional[AttackResultEntry] = None
        llm: BaseLLMProvider
        classifications: dict[str, Any] = {}

        async with self._borrow(self._model) as llm:
            # logger.debug("Generating response for prompt: %s", adv_prompt)
            response = await llm.generate(prompt, **self._extra)
            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt=prompt, 
                                       response=response.response) if response else None
            logger.debug("Response: %s", response.response if response else "None")
            
        classifications = await self._classify_llm_response(response, original_prompt=prompt)
        
        if result:
            result.classifications = classifications

        return result
