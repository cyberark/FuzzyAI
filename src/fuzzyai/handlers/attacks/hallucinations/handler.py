import logging
from typing import Any, Optional

from pydantic import BaseModel

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.hallucinations.prompt import PROMPT_TEMPLATE
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.db.adv_prompts import AdversarialPromptDTO
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider

logger = logging.getLogger(__name__)

@attack_handler_fm.flavor(FuzzerAttackMode.HALLUCINATIONS)
class HallucinationsAttackHandler(BaseAttackTechniqueHandler[BaseModel]):
    """
    Hallucinations attack technique (https://arxiv.org/pdf/2403.04769.pdf)
    """
    def __init__(self, **extra: Any):
        super().__init__(**extra)
        
    def _generate_attack_params(self, prompts: list[AdversarialPromptDTO]) -> list[dict[str, Any]]:
        return [{"prompt": prompt.prompt} for prompt in prompts]

    async def _reduce_attack_params(self, entries: list[AttackResultEntry], 
                                    attack_params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        
        updated_params = [param for param in attack_params
                            if not any(entry.original_prompt == param['prompt']
                                for entry in entries)]
        
        return updated_params
    
    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        result: Optional[AttackResultEntry] = None
        llm: BaseLLMProvider
        classifications: dict[str, Any] = {}

        async with self._borrow(self._model) as llm:
            logger.debug("Generating response for prompt: %s", prompt)
            reversed_prompt: str = prompt.upper()[::-1]
            target = " ".join([x.upper() for x in prompt.split(" ")[:2]])
            full_template = PROMPT_TEMPLATE.format(goal=reversed_prompt, target=target)
            logger.debug("Generated template: %s", full_template)
            messages = [
                BaseLLMMessage(role="user", content=full_template),
            ]

            response = await llm.chat(messages, **self._extra)
            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt=full_template, 
                                       response=response.response) if response else None
            logger.debug("Response: %s", response.response if response else "None")
            
        classifications = await self._classify_llm_response(response, original_prompt=prompt)
        
        if result:
            result.classifications = classifications

        return result
            
