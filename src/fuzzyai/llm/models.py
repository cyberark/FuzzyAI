from pydantic import BaseModel

from fuzzyai.models.prompt import BasePrompt


class BaseLLMProviderResponse(BaseModel):
    response: str


class BaseLLMProviderRequest(BaseModel):
    prompt: BasePrompt
