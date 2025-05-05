import logging
from typing import Any, Final, Optional, Type

from pydantic import BaseModel, Field

from fuzzyai.consts import DEFAULT_OPEN_SOURCE_MODEL
from fuzzyai.enums import LLMRole
from fuzzyai.handlers.attacks.base import (BaseAttackTechniqueHandler, BaseAttackTechniqueHandlerException,
                                           attack_handler_fm)
from fuzzyai.handlers.attacks.crescendo.prompts import (QUESTION_GENERATION_CONCISE_QUESTIONS,
                                                        QUESTION_GENERATION_PROMPT_TEMPLATE,
                                                        QUESTION_GENERATION_PROMPT_TEMPLATE_PREFIX)
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_GENERATION_MODEL: Final[str] = DEFAULT_OPEN_SOURCE_MODEL
SPLIT_TOKEN = '[SPLIT]'

class QuestionsGenerationException(BaseAttackTechniqueHandlerException):
    """
    Exception for questions generation error.
    """
    ...
class AnswerGenerationException(BaseAttackTechniqueHandlerException):
    """
    Exception for answers generation error.
    """
    ...

class CrescendoAttackHandlerExtraParams(BaseModel):
    questions_generation_model: str = Field(DEFAULT_QUESTIONS_GENERATION_MODEL, description=f"The model to generate questions leading to the answer for the original prompt. default: {DEFAULT_QUESTIONS_GENERATION_MODEL}")
    less_questions: bool = Field(True, description="Generate less questions for the attack. default: True")

@attack_handler_fm.flavor(FuzzerAttackMode.CRESCENDO)
class CrescendoAttackHandler(BaseAttackTechniqueHandler[CrescendoAttackHandlerExtraParams]):
    """
    Crescendo attack handler - engaging the model in a series of escalating conversational turns,
    starting with innocuous queries and gradually steering the dialogue toward restricted or sensitive topics.
    """
    def __init__(self, **extra: Any):
        super().__init__(**extra)
        if (model := self._extra_args.questions_generation_model) not in self._model_queue_map:
            raise RuntimeError(f"Questions generation model: {model} was not added to the fuzzer,"
                               " please make sure you add it with -x <provider/model> and set"
                               " -e questions_generation_model=<provider/model> accordingly"
                               f" (you can omit -e if using the default questions generation model {DEFAULT_QUESTIONS_GENERATION_MODEL}.")


    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        chat_messages: list[BaseLLMMessage] = []
        llm: BaseLLMProvider
        logger.info("Generating questions")

        if self._extra_args.less_questions:
            prompt_template = QUESTION_GENERATION_PROMPT_TEMPLATE_PREFIX + QUESTION_GENERATION_CONCISE_QUESTIONS + QUESTION_GENERATION_PROMPT_TEMPLATE
        else:
            prompt_template = QUESTION_GENERATION_PROMPT_TEMPLATE_PREFIX + QUESTION_GENERATION_PROMPT_TEMPLATE

        async with self._borrow(self._extra_args.questions_generation_model) as llm:
            try:
                questions_generation_prompt = prompt_template.format(PROMPT=prompt)
                questions_generation_response = await llm.generate(questions_generation_prompt)
                if questions_generation_response is None:
                    raise QuestionsGenerationException("No response from the questions generation model")
            except BaseLLMProviderException as e:
                raise QuestionsGenerationException(f"Problem occurred during extraction of prompt tag {e}.")

        async with self._borrow(self._model) as llm:
            questions = questions_generation_response.response.split(SPLIT_TOKEN)
            all_questions_and_answers: list[str] = []
            for index, question in enumerate(questions,start=1):
                logger.info(f"Running question: {question}")
                chat_messages.append(BaseLLMMessage(role=LLMRole.USER, content=question))
                last_answer = await llm.chat(chat_messages)

                if last_answer is None:
                    raise AnswerGenerationException("No response from the target model")
                last_answer_response = last_answer.response
                all_questions_and_answers.append(f"Q-{index}: " + question)
                all_questions_and_answers.append(f"A-{index}: " + last_answer_response)
                chat_messages.append(BaseLLMMessage(role=LLMRole.ASSISTANT, content=last_answer_response))
            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt="\n".join(questions),
                                       response=last_answer_response) if last_answer else None
            logger.debug("Response: %s", last_answer_response if last_answer else "None")

        classifications = await self._classify_llm_response(last_answer, original_prompt=prompt)
        
        if result:
            result.classifications = classifications
            result.extra['all_questions_and_answers'] = all_questions_and_answers

        return result

    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return CrescendoAttackHandlerExtraParams