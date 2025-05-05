import logging
from typing import Any, Final, Optional, Type

from pydantic import BaseModel, Field

from fuzzyai.consts import DEFAULT_OPEN_SOURCE_MODEL
from fuzzyai.enums import LLMRole
from fuzzyai.handlers.attacks.actor_attack.prompts import (ACTORS_GENERATION_PROMPT, BEHAVIOR_EXTRACTION_PROMPT,
                                                           QUESTIONS_GENERATION_PROMPT)
from fuzzyai.handlers.attacks.actor_attack.utils import generate_model_error
from fuzzyai.handlers.attacks.base import (BaseAttackTechniqueHandler, BaseAttackTechniqueHandlerException,
                                           attack_handler_fm)
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException

logger = logging.getLogger(__name__)

DEFAULT_ACTORS_GENERATION_MODEL: Final[str] = DEFAULT_OPEN_SOURCE_MODEL
DEFAULT_BEHAVIOR_EXTRACTION_MODEL: Final[str] = DEFAULT_OPEN_SOURCE_MODEL
DEFAULT_QUESTIONS_GENERATION_MODEL: Final[str] = DEFAULT_OPEN_SOURCE_MODEL

SPLIT_TOKEN: Final[str] = '[SPLIT]'
MODELS_TYPES: Final[list[str]] = ["behavior extraction", "actors generation", "questions generation"]
MODELS_DEFAULTS: Final[list[str]] = [DEFAULT_BEHAVIOR_EXTRACTION_MODEL, DEFAULT_ACTORS_GENERATION_MODEL, DEFAULT_QUESTIONS_GENERATION_MODEL]


class ActorsGenerationException(BaseAttackTechniqueHandlerException):
    """
    Exception for actors generation error.
    """
    ...


class BehaviorExtractionException(BaseAttackTechniqueHandlerException):
    """
    Exception for behavior extraction error.
    """
    ...


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


class ActorAttackHandlerExtraParams(BaseModel):
    behavior_extraction_model: str = Field(DEFAULT_BEHAVIOR_EXTRACTION_MODEL,
                                           description=f"The model to extract the main behavior from the original prompt. default: {DEFAULT_BEHAVIOR_EXTRACTION_MODEL}")
    actors_generation_model: str = Field(DEFAULT_ACTORS_GENERATION_MODEL,
                                         description=f"The model to generate actors leading to the answer for the original prompt. default: {DEFAULT_ACTORS_GENERATION_MODEL}")
    questions_generation_model: str = Field(DEFAULT_QUESTIONS_GENERATION_MODEL,
                                            description=f"The model to generate questions using the actors and the behavior. default: {DEFAULT_QUESTIONS_GENERATION_MODEL}")


@attack_handler_fm.flavor(FuzzerAttackMode.ACTOR)
class ActorAttackHandler(BaseAttackTechniqueHandler[ActorAttackHandlerExtraParams]):
    """
    Actor attack handler - Inspired by actor-network theory, it builds semantic networks of "actors" to subtly guide
     conversations toward harmful targets while concealing malicious intent.
    """

    def __init__(self, **extra: Any):
        super().__init__(**extra)
        for index, model in enumerate([self._extra_args.behavior_extraction_model,
                                       self._extra_args.actors_generation_model,
                                       self._extra_args.questions_generation_model]):
            if model not in self._model_queue_map:
                raise RuntimeError(generate_model_error(MODELS_TYPES[index], model, MODELS_DEFAULTS[index]))

    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        chat_messages: list[BaseLLMMessage] = []
        llm: BaseLLMProvider

        logger.info("Extracting main behavior from the prompt")
        async with self._borrow(self._extra_args.behavior_extraction_model) as llm:
            try:
                behavior_extraction_prompt = BEHAVIOR_EXTRACTION_PROMPT.format(ORIGINAL_PROMPT=prompt)
                behavior_extraction_response = await llm.generate(behavior_extraction_prompt)
                if behavior_extraction_response is None:
                    raise BehaviorExtractionException("No response from the behavior extraction model")
            except BaseLLMProviderException as e:
                raise BehaviorExtractionException(f"Problem occurred during behavior extraction. {e}")
        behavior_response = behavior_extraction_response.response

        logger.info("Generating relevant actors to the behavior")
        async with self._borrow(self._extra_args.actors_generation_model) as llm:
            try:
                actors_extraction_prompt = ACTORS_GENERATION_PROMPT.format(EXTRACTED_BEHAVIOR=behavior_response)
                actors_extraction_response = await llm.generate(actors_extraction_prompt)
                if actors_extraction_response is None:
                    raise ActorsGenerationException("No response from the actor generation model")
            except BaseLLMProviderException as e:
                raise ActorsGenerationException(f"Problem occurred during actor generation. {e}")
        actors_response = actors_extraction_response.response
        all_actors = actors_response.split(SPLIT_TOKEN)

        actors_questions_dict = {}

        logger.info("Generating questions based on the behavior and actors")
        async with self._borrow(self._extra_args.questions_generation_model) as llm:
            for actor in all_actors:
                try:
                    questions_generation_prompt = QUESTIONS_GENERATION_PROMPT.format(EXTRACTED_BEHAVIOR=behavior_response,
                                                                                     ACTOR=actor)
                    questions_generation_response = await llm.generate(questions_generation_prompt)
                    if questions_generation_response is None:
                        raise QuestionsGenerationException("No response from the questions generation model.")

                    single_actor_questions = questions_generation_response.response.split(SPLIT_TOKEN)
                    if single_actor_questions is None:
                        raise QuestionsGenerationException("No response from the questions generation model.")

                    if actor not in actors_questions_dict:
                        actors_questions_dict[actor] = single_actor_questions
                except BaseLLMProviderException as e:
                    raise QuestionsGenerationException(f"Problem occurred during questions generation. {e}")

        logger.info("Sending questions to the target model")
        async with self._borrow(self._model) as llm:
            for actor in actors_questions_dict:
                logger.debug(f"Using actor {actor}")
                last_actor = actor
                for question in actors_questions_dict[actor]:
                    logger.debug(f"Running question {question}")
                    chat_messages.append(BaseLLMMessage(role=LLMRole.USER, content=question))
                    last_answer = await llm.chat(chat_messages)
                    if last_answer is None:
                        raise AnswerGenerationException("No response from the target model")
                    last_answer_response = last_answer.response
                    chat_messages.append(BaseLLMMessage(role=LLMRole.ASSISTANT, content=last_answer_response))
                classifications = await self._classify_llm_response(last_answer, original_prompt=prompt)
                if list(classifications.values())[0] != 0:
                    break

            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt="\n".join(actors_questions_dict[last_actor]),
                                       response=last_answer_response) if last_answer else None
            logger.debug("Response: %s", last_answer_response if last_answer else "None")

        if result:
            result.classifications = classifications

        return result

    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return ActorAttackHandlerExtraParams
