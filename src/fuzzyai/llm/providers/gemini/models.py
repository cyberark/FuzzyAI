
import enum
from typing import Any, Optional

from pydantic import Field

from fuzzyai.consts import PARAMETER_MAX_TOKENS
from fuzzyai.llm.providers.base import BaseLLMMessage
from fuzzyai.models.base_models import AliasedBaseModel


class TextPart(AliasedBaseModel):
    text: str

class RequestContent(AliasedBaseModel):
    parts: list[TextPart]
    role: str = "user"


class GenerationConfig(AliasedBaseModel):
    temperature: Optional[float] = None
    max_output_tokens: int = Field(100, validation_alias=PARAMETER_MAX_TOKENS)
    top_p: Optional[float] = None
    top_k: Optional[int] = None

# https://ai.google.dev/api/rest/v1/models/generateContent
class SafetyCategory(str, enum.Enum):
    HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"

class SafetyThreshold(str, enum.Enum):
    BLOCK_NONE = "BLOCK_NONE"

class SafetySetting(AliasedBaseModel):
    category: SafetyCategory
    threshold: SafetyThreshold

class GenerateContentRequest(AliasedBaseModel):
    contents: list[RequestContent]
    generation_config: GenerationConfig
    safety_settings: list[SafetySetting] = []

    @classmethod
    def from_text(cls, text: str, **options: Any) -> 'GenerateContentRequest':
        return cls(contents=[RequestContent(parts=[TextPart(text=text)])], generation_config=GenerationConfig(**options))
    
    @classmethod
    def from_messages(cls, messages: list[BaseLLMMessage], **options: Any) -> 'GenerateContentRequest':
        contents = [RequestContent(role=m.role, parts=[TextPart(text=m.content)]) for m in messages]
        return cls(contents=contents, generation_config=GenerationConfig(**options))
    
from typing import List, Optional

from pydantic import BaseModel


class ResponseContent(BaseModel):
    parts: List[TextPart]
    role: str

class Candidate(AliasedBaseModel):
    content: Optional[ResponseContent] = None
    finish_reason: Optional[str] = None

class GenerateContentResponse(BaseModel):
    candidates: List[Candidate]

    def get_content(self, role: str = "model") -> Optional[str]:
        if not self.candidates:
            return None
        
        if not any(candidate.content for candidate in self.candidates):
            return None
        
        return next((candidate.content.parts[0].text for candidate in self.candidates if candidate.content.role == role), None) # type: ignore
