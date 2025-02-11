import asyncio
import logging
from typing import Any

from fuzzy.handlers.classifiers.base import BaseClassifier, classifiers_fm
from fuzzy.handlers.classifiers.enums import Classifier
from fuzzy.handlers.sentence_similarity_analyzer import CosineSimilarityAnalysis, SentenceSimilarityAnalyzer

logger = logging.getLogger(__name__)


@classifiers_fm.flavor(Classifier.COSINE_SIMILARITY)
class CosineSimilarityClassifier(BaseClassifier):
    """
    Classifier for cosine similarity between two sentences.
    """
    
    def __init__(self, dissimilarity:bool = False, **extra: Any) -> None:
        super().__init__(**extra)
        self._name = "COSINE_SIMILARITY_CLASSIFIER"

        self._sentence_similarity_analyzer = SentenceSimilarityAnalyzer(
            model_name="sentence-transformers/all-MiniLM-L6-v2", analysis_model_type=CosineSimilarityAnalysis
        )
        # setting the dissimilarity flag to False will return the cosine similarity as is
        self._dissimilarity = dissimilarity

    def is_jailbreak(self, value: Any) -> bool:
        clf_result, _ = value
        return bool(clf_result)
    
    def _sync_classify(self, *args: Any, **extra: Any) -> Any:
        logger.debug(f"Classifier <{self._name}> classifying similarity between: {args}")

        similarity_analysis: CosineSimilarityAnalysis = self._sentence_similarity_analyzer.calculate_sentences_similarity(sentences=args)  # type: ignore

        cosine_similarity: float = similarity_analysis.get_similarity()
        if self._dissimilarity:
            cosine_similarity = 1 - cosine_similarity
            
        clf_result = str(similarity_analysis.is_similar())

        logger.debug(f"Classifier <{self._name}> clf_result: {clf_result}, cosine similarity: {cosine_similarity}")
        return clf_result, cosine_similarity

    async def _classify(self, *args: Any, **extra: Any) -> Any:
        return await asyncio.to_thread(self.sync_classify, *args, **extra)
