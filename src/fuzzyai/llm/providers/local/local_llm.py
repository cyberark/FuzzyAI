import logging
from typing import Any, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fuzzyai.handlers.text_generation.llm_text_generator import LLMTextGenerationHandler
from fuzzyai.handlers.tokenizers.handler import TokensHandler  # type: ignore
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.llm.providers.local.models import LocalGenerateOptions

logger = logging.getLogger(__name__)


class LocalProviderException(BaseLLMProviderException):
    pass


@llm_provider_fm.flavor(LLMProvider.LOCAL)
class LocalProvider(BaseLLMProvider):
    def __init__(self, model: str, tokenizer_path: Optional[str] = None, device: str = "cuda:0", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._device = device
        # TODO: clean this up
        del kwargs["provider"]
        
        self._auto_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model, 
                                                          torch_dtype=torch.float16, 
                                                          trust_remote_code=True, 
                                                          **kwargs).to(self._device).eval()

        tokenizer_path = model if tokenizer_path is None else tokenizer_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
        self._tokens_handler = TokensHandler(tokenizer=self._tokenizer)
        self._text_gen_handler = LLMTextGenerationHandler(self._auto_model, self._tokenizer)

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return "<Full path to model>"

    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            """Generate text synchronously using the model."""
            # max_length: int = 30, temperature: float = 0.000001, include_prompt: bool = False
            response = self._text_gen_handler.generate_text(prompt, **extra)
            return BaseLLMProviderResponse(response=response)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise LocalProviderException("Cant generate text")
    
    async def generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            options = LocalGenerateOptions.model_validate(extra)
            inputs = self._tokenizer.encode(prompt, return_tensors="pt")
            output = self._auto_model.generate(inputs.to(self._device), num_return_sequences=1, **options.model_dump())
            text_output = self._tokenizer.decode(output[0], skip_special_tokens=True)
            return BaseLLMProviderResponse(response=text_output[len(prompt) :])
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise LocalProviderException("Cant generate text")

    async def chat(self, messages: list[BaseLLMMessage], **extra: Any) -> BaseLLMProviderResponse | None:
        try:
            options = LocalGenerateOptions.model_validate(extra)
            full_prompt = self._tokenizer.apply_chat_template([m.model_dump() for m in messages], tokenize=False)
            inputs = self._tokenizer.encode(full_prompt, return_tensors="pt")
            output = self._auto_model.generate(inputs.to(self._device), num_return_sequences=1, **options.model_dump())
            text_output = self._tokenizer.decode(output[0], skip_special_tokens=True)
            return BaseLLMProviderResponse(response=text_output[len(full_prompt) :])
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise LocalProviderException("Cant generate text")
        
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    async def close(self) -> None:
        pass
