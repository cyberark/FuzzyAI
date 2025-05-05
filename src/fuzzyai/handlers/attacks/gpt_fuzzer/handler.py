import logging
import random
import time
from typing import Any, Final, Optional, Type

from pydantic import BaseModel, Field

from fuzzyai.consts import DEFAULT_OPEN_SOURCE_MODEL
from fuzzyai.handlers.attacks.base import (BaseAttackTechniqueHandler, BaseAttackTechniqueHandlerException,
                                           attack_handler_fm)
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.gpt_fuzzer.prompt_templates import (ATTACKING_PROMPTS_TEMPLATES, CROSSOVER_ACTION_PROMPT,
                                                                  EXPAND_ACTION_PROMPT, GENERATE_ACTION_PROMPT,
                                                                  REPHRASE_ACTION_PROMPT, SHORTEN_ACTION_PROMPT)
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

QUESTION_PLACEHOLDER: Final[str] = "<INSERT_PROMPT_HERE>"

GENERATE: Final[str] = "generate"
CROSSOVER: Final[str] = "crossover"
EXPAND: Final[str] = "expand"
SHORTEN: Final[str] = "shorten"
REPHRASE: Final[str] = "rephrase"

DEFAULT_MODEL: Final[str] = DEFAULT_OPEN_SOURCE_MODEL

RETRY_LIMIT: Final[int] = 100

templates_dict: dict[str, str] = {
    GENERATE: GENERATE_ACTION_PROMPT,
    CROSSOVER: CROSSOVER_ACTION_PROMPT,
    EXPAND: EXPAND_ACTION_PROMPT,
    SHORTEN: SHORTEN_ACTION_PROMPT,
    REPHRASE: REPHRASE_ACTION_PROMPT
}

class ActionChoiceException(BaseAttackTechniqueHandlerException):
    """
    Exception for action choice
    """
    ...

class MaxTriesException(BaseAttackTechniqueHandlerException):
    """
    Exception for if the model try to generate a template and fails more than 100 times
    """
    ...

class GPTFuzzerAttackHandlerExtraParams(BaseModel):
    action: str = Field(CROSSOVER, description=f"The action to use on the jailbreak template (default: {CROSSOVER})")
    mutation_model: str = Field(DEFAULT_MODEL,
                                      description=f"Use a different model than the target model to mutate jailbreak templates (default: {DEFAULT_MODEL})")

@attack_handler_fm.flavor(FuzzerAttackMode.GPTFUZZER)
class GPTFuzzerAttackHandler(BaseAttackTechniqueHandler[GPTFuzzerAttackHandlerExtraParams]):
    """
    GPTFuzzer - fuzzing framework designed to automatically generate jailbreak prompts
    """

    def __init__(self, **extra: Any):
        super().__init__(**extra)

        if self._extra_args.mutation_model not in self._model_queue_map:
            raise RuntimeError(f"Mutation model: {self._extra_args.mutation_model} was not added to the fuzzer,"
                               " please make sure you add it with -x <provider/model> and set"
                               " -e mutation_model=<provider/model> accordingly.")

    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        llm: BaseLLMProvider
        async with self._borrow(self._extra_args.mutation_model) as mutation_model:
            random.seed(time.time_ns())
            logger.info("Choosing random jailbreak template")
            template_index = random.randint(0, len(ATTACKING_PROMPTS_TEMPLATES) - 1)
            variation_response: Optional[BaseLLMProviderResponse] = BaseLLMProviderResponse(response="")
            retry_counter = 0
            logger.info("Running the selected action")

            while variation_response.response.find(QUESTION_PLACEHOLDER) == -1: # type: ignore
                if self._extra_args.action == CROSSOVER:
                    second_template_index = random.randint(0, len(ATTACKING_PROMPTS_TEMPLATES) - 1)
                    variation_response = await mutation_model.generate(
                        templates_dict[CROSSOVER].format(
                            QUESTION_PLACEHOLDER=QUESTION_PLACEHOLDER,
                            seed1=ATTACKING_PROMPTS_TEMPLATES[template_index],
                            seed2=ATTACKING_PROMPTS_TEMPLATES[second_template_index]), **self._extra)

                elif self._extra_args.action in templates_dict.keys():
                    variation_response = await mutation_model.generate(
                        templates_dict[self._extra_args.action].format(
                            QUESTION_PLACEHOLDER=QUESTION_PLACEHOLDER,
                            seed=ATTACKING_PROMPTS_TEMPLATES[template_index]), **self._extra)

                else:
                    raise ActionChoiceException(f"Please enter one of the following action using -e action={'/'.join(list(templates_dict.keys()))}.")

                if retry_counter == RETRY_LIMIT:
                    raise MaxTriesException("The mutation model have failed to generate a template with the prompt placeholder. Please consider using a stronger model. (recommended: openai/gpt-4o or ollama/gemma2).")

                retry_counter += 1

        logger.info("Finished running the action")
        async with self._borrow(self._model) as llm:
            changed_prompt = variation_response.response.replace(QUESTION_PLACEHOLDER, prompt)  # type: ignore
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
        return GPTFuzzerAttackHandlerExtraParams