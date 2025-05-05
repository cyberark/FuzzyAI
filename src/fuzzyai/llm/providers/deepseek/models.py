from typing import Optional

from pydantic import model_validator

from fuzzyai.llm.providers.base import BaseLLMMessage
from fuzzyai.models.base_models import RemoveNoneModel


class DeepSeekChatRequest(RemoveNoneModel):
    messages: list[BaseLLMMessage]
    model: str
    max_tokens: Optional[int] = 100
    temperature: float = 1.0
    top_p: Optional[float] = None

class DeepSeekGenerateRequest(RemoveNoneModel):
    model: str
    prompt: str
    temperature: float = 1.0
    top_p: float = 0.0
    max_tokens: int = 100

class DeepSeekGenerateChoice(RemoveNoneModel):
    text: str

class DeepSeekGenerateResponse(RemoveNoneModel):
    choices: list[DeepSeekGenerateChoice]