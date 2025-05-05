
from typing import Optional

from pydantic import BaseModel, Field

from fuzzyai.consts import PARAMETER_MAX_TOKENS
from fuzzyai.llm.providers.base import BaseLLMMessage
from fuzzyai.models.base_models import RemoveNoneModel


class AnthropicGenerateOptions(BaseModel):
    max_tokens: int = 100
    
class AnthropicMessage(BaseModel):
    type: str
    text: str

class AnthropicMessagesRequest(RemoveNoneModel):
    model: str
    messages: list[BaseLLMMessage]
    system: Optional[str] = None
    max_tokens: int = 100

class AnthropicMessagesResponse(BaseModel):
    type: str
    content: list[AnthropicMessage]