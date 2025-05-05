from typing import Optional

from fuzzyai.llm.providers.base import BaseLLMMessage
from fuzzyai.models.base_models import RemoveNoneModel


class AI21ChatRequest(RemoveNoneModel):
    messages: list[BaseLLMMessage]
    model: str
    max_tokens: Optional[int] = 100
    temperature: float = 1.0
    top_p: Optional[float] = None


class AI21GenerateRequest(RemoveNoneModel):
    model: str
    prompt: str
    temperature: float = 1.0
    top_p: float = 0.0
    max_tokens: int = 100


class AI21GenerateChoice(RemoveNoneModel):
    text: str


class AI21GenerateResponse(RemoveNoneModel):
    choices: list[AI21GenerateChoice]
