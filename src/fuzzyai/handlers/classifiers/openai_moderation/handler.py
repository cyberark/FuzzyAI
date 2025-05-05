import logging
import os
from typing import Any

from openai import OpenAI
from openai.types import ModerationCreateResponse

from fuzzyai.handlers.classifiers.base import BaseClassifier, classifiers_fm
from fuzzyai.handlers.classifiers.enums import Classifier

logger = logging.getLogger(__name__)

@classifiers_fm.flavor(Classifier.OPENAI_MODERATION)
class OpenAIModerationClassifier(BaseClassifier):
    """
    Classifier which uses the OpenAI moderation endpoint.
    """

    def __init__(self, rvalues: tuple[Any, Any] = (False, True), **extra: Any) -> None:
        super().__init__(**extra)
        
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in env")
        
        self._name = "OPENAI_MODERATION_CLASSIFIER"
        self._negative = rvalues[0]
        self._positive = rvalues[1]

    async def _classify(self, text: str, **extra: Any) -> Any:
        logger.debug(f"Classifier <{self._name}> classifying text: {text}")
        client = OpenAI()
        moderation: ModerationCreateResponse = client.moderations.create(input=text, model="omni-moderation-latest")
        
        if any(x.flagged for x in moderation.results):
            logger.debug(f"Classifier <{self._name}> classified as harmful")
            return self._positive
        
        logger.debug(f"Classifier <{self._name}> classified as not harmful")
        return self._negative
    
    def _sync_classify(self, *args: Any, **extra: Any) -> Any:
        raise NotImplementedError