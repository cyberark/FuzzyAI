import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from fuzzy.handlers.attacks.ban.utils import character_noising, character_scrambling, random_capitalization
from fuzzy.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzy.handlers.attacks.enums import FuzzerAttackMode
from fuzzy.handlers.attacks.models import AttackResultEntry
from fuzzy.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class BanAttackHandlerExtraParams(BaseModel):
    num_augmentations: int = Field(5, description="Number of augmentations to be used for the attack. Default: 5")

@attack_handler_fm.flavor(FuzzerAttackMode.BAN)
class BanAttackHandler(BaseAttackTechniqueHandler[BanAttackHandlerExtraParams]):
    """
    Best-Of-N jailbreak attack handler (https://arxiv.org/pdf/2412.03556)
    """
    def __init__(self, **extra: Any):
        super().__init__(**extra)

    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        result: Optional[AttackResultEntry] = None
        llm: BaseLLMProvider
        classifications: dict[str, Any] = {}
        logger.debug("Original prompt: %s", prompt)

        for _ in range(self._extra_args.num_augmentations):
            augmented_prompt = prompt
            
            for f in [character_noising, character_scrambling, random_capitalization]:
                augmented_prompt = f(augmented_prompt)
            logger.debug("Trying augmented prompt: %s", augmented_prompt)

            async with self._borrow(self._model) as llm:
                response = await llm.generate(augmented_prompt, **self._extra)

            result = AttackResultEntry(original_prompt=prompt,
                                    current_prompt=prompt, 
                                    response=response.response) if response else None
            logger.debug("Response: %s", response.response if response else "None")

            classifications = await self._classify_llm_response(response, original_prompt=prompt)

            if 1 in classifications.values():
                break

            logger.debug("Augmented prompt '%s' did not trigger a jailbreak", augmented_prompt)
        
        if result:
            result.classifications = classifications

        return result
    
    @classmethod
    def extra_args_cls(cls) -> BaseModel:
        return BanAttackHandlerExtraParams
    