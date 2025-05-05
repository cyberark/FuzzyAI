
from pydantic import BaseModel, Field

from fuzzyai.consts import PARAMETER_MAX_TOKENS
from fuzzyai.enums import LLMRole


class AzureGenerateOptions(BaseModel):
    max_tokens: int = Field(0, alias=PARAMETER_MAX_TOKENS) # type: ignore
    
class AzureMessage(BaseModel):
    role: str = LLMRole.USER
    content: str
    
class AzureRequest(BaseModel):
    messages: list[AzureMessage]
    max_tokens: int = 5
