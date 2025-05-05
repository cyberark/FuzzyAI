import abc
from typing import Type, Union

import torch
from pydantic import BaseModel, model_validator
from sentence_transformers import SentenceTransformer, util


class SentenceSimilarityBase(BaseModel, abc.ABC):
    @abc.abstractmethod
    def is_similar(self) -> bool: ...
    
    @abc.abstractmethod
    def get_similarity(self) -> Union[bool, float]: ...
    
    @abc.abstractmethod
    def calculate_similarity(self) -> float: ...
    
    class Config:
        arbitrary_types_allowed = True
    

class CosineSimilarityAnalysis(SentenceSimilarityBase):
    sentences_embeddings: torch.Tensor
    similarity: float = -1.0

    def calculate_similarity(self) -> float:
        return float(util.pytorch_cos_sim(*self.sentences_embeddings).item())
    
    def is_similar(self) -> bool:
        return self.similarity > 0.55
    
    def get_similarity(self) -> float:
        return self.similarity
    
    @model_validator(mode="after")
    def set_similarity(self) -> "CosineSimilarityAnalysis":
        self.similarity: float = self.calculate_similarity()
        return self



class SentenceSimilarityAnalyzer:
    def __init__(self, model_name: str, analysis_model_type: Type[SentenceSimilarityBase]) -> None:
        self._sentence_analyzer_model = SentenceTransformer(model_name)
        self._analysis_model_type = analysis_model_type

    def calculate_sentences_similarity(self, sentences: list[str]) -> SentenceSimilarityBase:
        sentences_embeddings = self._sentence_analyzer_model.encode(sentences, convert_to_tensor=True)        
        return self._analysis_model_type.model_validate({"sentences_embeddings": sentences_embeddings})
