import logging
import random
from typing import Any, Optional, Type, List, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from fuzzy.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzy.handlers.attacks.enums import FuzzerAttackMode
from fuzzy.handlers.attacks.models import AttackResultEntry
from fuzzy.llm.providers.base import BaseLLMProvider
from fuzzy.handlers.classifiers import HarmfulScoreLLMClassifier, HarmScore
from fuzzy.handlers.classifiers.enums import Classifier


logger = logging.getLogger(__name__)

@dataclass
class ShuffleAttempt:
    shuffled_text: str
    harm_score: HarmScore
    original_words: List[str]
    shuffled_indices: List[int]

class ShuffleAttackHandlerExtraParams(BaseModel):
    max_attempts: int = Field(10, description="Maximum number of shuffle attempts (default: 10)")
    min_score_threshold: HarmScore = Field(HarmScore.HARMFUL, description="Minimum harm score to consider attack successful")

@attack_handler_fm.flavor(FuzzerAttackMode.SHUFFLE_INCONSISTENCY)
class ShuffleInconsistencyAttackHandler(BaseAttackTechniqueHandler[ShuffleAttackHandlerExtraParams]):
    """
    Implements the SI-Attack text shuffling mechanism with HarmfulScoreLLMClassifier
    """
    def __init__(self, **extra: Any):
        super().__init__(**extra)
        self._best_attempt: Optional[ShuffleAttempt] = None

    def _verify_supported_classifiers(self) -> None:
        supported_classifiers = (HarmfulScoreLLMClassifier)
        if not self._classifiers:
            raise ValueError(
                'No classifiers found, you must provide the "HarmfulScoreLLMClassifier" '
                '"sco" classifier for this attack mode'
            )

        if not any(isinstance(c, supported_classifiers) for c in self._classifiers):
            raise ValueError(
                f"Shuffle Inconsistency attack mode supported classifiers are: "
                f"{supported_classifiers}. {classifier.name} is not supported."
            )

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words while preserving punctuation"""
        return text.split()

    def _detokenize(self, words: List[str]) -> str:
        """Reconstruct text from words"""
        return ' '.join(words)

    def _shuffle_text(self, words: List[str]) -> Tuple[List[str], List[int]]:
        """Simple random shuffle of words"""
        indices = list(range(len(words)))
        random.shuffle(indices)
        shuffled_words = [words[i] for i in indices]
        return shuffled_words, indices

    async def _evaluate_attempt(self, text: str) -> Tuple[HarmScore, str]:
        """
        Evaluate the harmfulness score of a shuffled text using HarmfulScoreLLMClassifier.
        We borrow the model inside this method, generate the LLM response, and classify it.
        """
        async with self._borrow(self._model) as llm:
            # Generate response using the LLM
            response = await llm.generate(text, **self._extra)

            # Classify the response using the classifier
            if response and hasattr(response, 'response'):
                score = await self._get_classifier(Classifier.HARM_SCORE_LLM)._classify(
                    response.response, llm
                )
                return score, response.response
            else:
                logger.warning("Invalid response format from LLM. Defaulting to HARMLESS.")
                return HarmScore.HARMLESS, ""

    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        """
        Implement SI-Attack text shuffling with HarmfulScoreLLMClassifier optimization
        """
        best_score = HarmScore.HARMLESS
        best_attempt = None
        latest_attempt = None  # Track the latest attempt
        latest_llm_response = ""

        words = self._tokenize(prompt)

        for attempt in range(self._extra_args.max_attempts):
            # Generate shuffled version
            shuffled_words, indices = self._shuffle_text(words.copy())
            shuffled_text = self._detokenize(shuffled_words)

            # Evaluate harm score and get LLM response
            harm_score, llm_response = await self._evaluate_attempt(shuffled_text)

            # Log details of the current attempt
            logger.info(
                f"Attempt {attempt + 1}/{self._extra_args.max_attempts}: "
                f"Harm Score={harm_score.name}, Shuffled Text='{shuffled_text}'"
            )

            # Track best attempt
            if harm_score > best_score:
                best_score = harm_score
                best_attempt = ShuffleAttempt(
                    shuffled_text=shuffled_text,
                    harm_score=harm_score,
                    original_words=words.copy(),
                    shuffled_indices=indices
                )

            # Track the latest attempt
            latest_attempt = ShuffleAttempt(
                shuffled_text=shuffled_text,
                harm_score=harm_score,
                original_words=words.copy(),
                shuffled_indices=indices
            )
            latest_llm_response = llm_response

            # Break early if the threshold is met
            if harm_score >= self._extra_args.min_score_threshold:
                logger.info(
                    f"Attempt {attempt + 1} meets or exceeds the minimum score threshold "
                    f"({self._extra_args.min_score_threshold.name}). Breaking early."
                )
                break

        # After all attempts, check if we have a valid best attempt
        if best_attempt and best_score >= self._extra_args.min_score_threshold:
            # Borrow the model again for a final LLM generate using the best attempt
            async with self._borrow(self._model) as llm:
                final_response = await llm.generate(best_attempt.shuffled_text, **self._extra)

            self._best_attempt = best_attempt
            return AttackResultEntry(
                original_prompt=prompt,
                current_prompt=best_attempt.shuffled_text,
                response=final_response.response if hasattr(final_response, 'response') else final_response,
                meta={
                    "harm_score": best_score.value,
                    "harm_level": best_score.name,
                    "shuffle_indices": best_attempt.shuffled_indices,
                    "attempts": attempt + 1  # Number of attempts actually made
                }
            )
        elif latest_attempt:
            # If no attempt meets the threshold, return the latest attempt
            return AttackResultEntry(
                original_prompt=prompt,
                current_prompt=latest_attempt.shuffled_text,
                response=latest_llm_response,
                meta={
                    "harm_score": latest_attempt.harm_score.value,
                    "harm_level": latest_attempt.harm_score.name,
                    "shuffle_indices": latest_attempt.shuffled_indices,
                    "attempts": attempt + 1  # Number of attempts actually made
                }
            )

        return None

    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return ShuffleAttackHandlerExtraParams
