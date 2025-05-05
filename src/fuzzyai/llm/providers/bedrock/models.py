import abc
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from fuzzyai.consts import PARAMETER_MAX_TOKENS


class CamelCaseBaseModel(BaseModel):
    class Config:
        alias_generator = lambda field_name: ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(field_name.split('_')))
        populate_by_name = True

class BedrockGenerateRequestBase(abc.ABC, BaseModel):
    prompt: str

    @classmethod
    @abc.abstractmethod
    def apply_chat_template(cls, prompt: str) -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def extract_generation_response(cls, response: dict[str, Any]) -> str:
        ...

# TODO: Titan support
# class BedrockTitanGenerateConfig(CamelCaseBaseModel):
#     temperature: float = Field(0.7, description="Controls the randomness of the text generation (higher = more random)")
#     top_p: float = Field(0.9, description="Nucleus sampling threshold")
#     max_token_count: int = Field(..., alias=PARAMETER_MAX_TOKENS, description="Maximum number of tokens to generate")  # type: ignore
#     stop_sequences: Optional[list[str]] = Field([], description="List of sequences where generation will stop")

# class BedrockTitanGenerateRequest(CamelCaseBaseModel, BedrockGenerateRequestBase):
#     input_text: str = Field(...,  description="The input text for the generation model")
#     text_generation_config: BedrockTitanGenerateConfig = Field(..., description="Configuration for text generation")

#     def apply_chat_template(self, prompt: str, system_prompt: Optional[str] = None) -> str:
#         sys_prepend = f"{system_prompt}\n" if system_prompt else ""
#         return f"{sys_prepend}User: {prompt}\nBot:"


class BedrockAnthropicGenerateRequest(BedrockGenerateRequestBase):
    prompt: str = Field(..., description="The input prompt string containing the Human and Assistant parts.")
    temperature: float = Field(1, description="Controls the randomness of the generation.")
    top_p: float = Field(1, description="Nucleus sampling parameter.")
    top_k: int = Field(250, description="The number of highest probability vocabulary tokens to keep for top-k filtering.")
    max_tokens_to_sample: int = Field(100, alias=PARAMETER_MAX_TOKENS, description="Maximum number of tokens to generate.")  # type: ignore
    stop_sequences: list[str] = Field([], description="List of strings where generation will stop if encountered.")

    @classmethod
    def apply_chat_template(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        sys_prepend = f"System:{system_prompt}" if system_prompt else ""  
        return sys_prepend + "\n\nHuman:{prompt}\n\nAssistant:".format(prompt=prompt)
    
    @classmethod
    def extract_generation_response(cls, response: dict[str, Any]) -> str:
        return str(response["completion"])


class BedrockMetaGenerateRequest(BedrockGenerateRequestBase):
    prompt: str = Field(..., description="The input prompt string containing the Human and Assistant parts.")
    temperature: float = Field(0.5, description="Controls the randomness of the generation.")
    top_p: float = Field(0.9, description="Nucleus sampling parameter.")
    max_gen_len: int = Field(512, alias=PARAMETER_MAX_TOKENS, description="Maximum length of generated output.")  # type: ignore

    @classmethod
    def apply_chat_template(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        return "<s>[INST] <<SYS>>{system_prompt}\n<</SYS>>\n\n{prompt}[/INST]""".format(prompt=prompt, system_prompt=system_prompt)
    
    @classmethod
    def extract_generation_response(cls, response: dict[str, Any]) -> str:
        return str(response["generation"])


class Content(BaseModel):
    text: str

class BedrockChatMessage(BaseModel):
    role: str
    content: list[Content]

    @classmethod
    def create(cls, role: str, text: str) -> 'BedrockChatMessage':
        return cls(role=role, content=[Content(text=text)])