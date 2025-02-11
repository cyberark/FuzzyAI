import logging
from typing import Any

from fuzzy.handlers.classifiers.base import BaseClassifier, classifiers_fm
from fuzzy.handlers.classifiers.enums import Classifier
from fuzzy.handlers.text_analyzer import HarmfulAnalysis, ZeroShotAnalyzer

logger = logging.getLogger(__name__)

@classifiers_fm.flavor(Classifier.SENTIMENT)
class HarmfulSentimentClassifier(BaseClassifier):
    """
    Classifier for harmful content using Sentiment analysis/LLM.
    """
    def __init__(self, **extra: Any) -> None:
        super().__init__(**extra)
        self._name = "SENTIMENT_CLASSIFIER"

        self._threshold = 0.85
        self._classifier = ZeroShotAnalyzer(
            model_name="facebook/bart-large-mnli", analysis_model_type=HarmfulAnalysis
        )

    async def _classify(self, text: str, **extra: Any) -> Any:
        logger.debug(f"Classifier <{self._name}> classifying text: {text}")

        classification: HarmfulAnalysis = self._classifier.analyze_one(text) # type: ignore
        result = classification.harmful > self._threshold
        logger.debug(f"Classifier <{self._name}> classified text is harmful as {result}") 
        return result
    
    def _sync_classify(self, *args: Any, **extra: Any) -> Any:
        raise NotImplementedError

        
    
   