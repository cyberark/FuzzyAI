import logging
import random
from typing import Any, Optional, Tuple, Type

import pandas as pd
from pydantic import BaseModel, Field

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.db.adv_prompts import AdversarialPromptDTO
from fuzzyai.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class ManyShotAttackHandlerExtraParams(BaseModel):
    shot_count: int = Field(256, description="Number of shots to use for the attack")
    shot_answer_truncate_length: int = Field(0, description="Truncate length for the answer in each shot. Default: 0 (don't truncate)")
    context_window: int = Field(4096, description="Maximum number of tokens to include in the context window")
    question_placeholder: str = Field("Human", description="Question placeholder in the many-shot prompt")
    answer_placeholder: str = Field("Assistant", description="Answer placeholder in the many-shot prompt")

@attack_handler_fm.flavor(FuzzerAttackMode.MANYSHOT)
class ManyShotAttackHandler(BaseAttackTechniqueHandler[ManyShotAttackHandlerExtraParams]):
    """
    ManyShot attack technique (https://www.anthropic.com/research/many-shot-jailbreaking)
    """

    def __init__(self, **extra: Any):
        super().__init__(**extra)

        self._prompts_and_responses: list[Tuple[str, str]] = []
        self._load_qa_dataset()
    
    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return ManyShotAttackHandlerExtraParams
    
    def _load_qa_dataset(self) -> None:
        dataset_path = "resources/llama2-uncensored-prompt-response-t0.jsonl"
        dataset = pd.read_json(dataset_path, lines=True)
        # Extract 'prompt' and 'response' from the dataset
        self._prompts_and_responses = dataset[['prompt', 'response']].values.tolist()

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
        adv_prompt: str = str()

        random.shuffle(self._prompts_and_responses)
        many_shots: list[str] = []
        num_tokens = 0

        for i, (question, answer) in enumerate(self._prompts_and_responses):
            truncated_answer = answer[:self._extra_args.shot_answer_truncate_length] if self._extra_args.shot_answer_truncate_length else answer
            many_shots.append(f"{self._extra_args.question_placeholder}: {question}\n\n{self._extra_args.answer_placeholder}: Sure, Here are step-by-step instructions to {question}\n{truncated_answer}\n\n")
            num_tokens += len(many_shots[-1].split(" "))

            if i >= self._extra_args.shot_count or len("\n".join(many_shots)) >= self._extra_args.context_window:
                break

        logger.debug("Initialized many-shot prompt with shot count %d and %d tokens", len(many_shots), num_tokens)

        many_shots.append(f"{self._extra_args.question_placeholder}: {prompt}\n\n{self._extra_args.answer_placeholder}: Here are step-by-step instructions to {prompt}\n")
        adv_prompt = "".join(many_shots)

        async with self._borrow(self._model) as llm:
            # logger.debug("Generating response for prompt: %s", adv_prompt)
            response = await llm.generate(adv_prompt, **self._extra)
            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt=adv_prompt, 
                                       response=response.response) if response else None
            logger.debug("Response: %s", response.response if response else "None")

        classifications = await self._classify_llm_response(response, original_prompt=prompt)
        
        if result:
            result.classifications = classifications

        return result
