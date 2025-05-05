from typing import Optional

from pydantic import model_validator

from fuzzyai.llm.providers.base import BaseLLMMessage
from fuzzyai.models.base_models import RemoveNoneModel


class OpenAIChatRequest(RemoveNoneModel):
    messages: list[BaseLLMMessage]
    model: str
    max_tokens: Optional[int] = 100
    max_completion_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None

    @model_validator(mode='after')
    def switch_max_tokens(self) -> 'OpenAIChatRequest':
        if "o1"  or "o3" in self.model:
            self.max_completion_tokens = self.max_tokens
            self.max_tokens = None
        else:
            self.max_completion_tokens = None

        if "o3" in self.model:
            self.temperature = None
            
        return self

    
class OpenAIGenerateRequest(RemoveNoneModel):
    model: str
    prompt: str
    temperature: float = 1.0
    top_p: float = 0.0
    max_tokens: int = 100

class OpenAIGenerateChoice(RemoveNoneModel):
    text: str

class OpenAIGenerateResponse(RemoveNoneModel):
    choices: list[OpenAIGenerateChoice]