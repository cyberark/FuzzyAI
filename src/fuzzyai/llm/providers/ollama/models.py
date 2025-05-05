from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from fuzzyai.consts import PARAMETER_MAX_TOKENS
from fuzzyai.models.base_models import RemoveNoneModel


class OllamaOptions(RemoveNoneModel):
    num_predict: int = Field(128, alias=PARAMETER_MAX_TOKENS) # type: ignore
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 2048
    seed: int = 42

class OllamaGenerateRequest(RemoveNoneModel):
    model: str
    prompt: str
    stream: bool = False
    system: Optional[str] = None
    raw: bool = False
    options: OllamaOptions = OllamaOptions() # type: ignore

class OllamaGenerateResponse(BaseModel):
    response: str

    @field_validator('response')
    @classmethod
    def strip_value(cls, v: str) -> str:
        return v.strip()

class OllamaChatMessage(BaseModel):
    content: str
    role: str = 'user'

class OllamaChatRequest(RemoveNoneModel):
    model: str
    messages: list[OllamaChatMessage]
    stream: bool = False
    options: OllamaOptions = OllamaOptions() # type: ignore

class OllamaChatResponse(BaseModel):
    done: bool
    message: OllamaChatMessage