import logging
import os
from typing import Any, Optional, Union

import aiohttp
import backoff
import requests
import tiktoken

from fuzzyai.enums import EnvironmentVariables, LLMRole
from fuzzyai.handlers.tokenizers.handler import TokensHandler  # type: ignore
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import (BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException,
                                        BaseLLMProviderRateLimitException, llm_provider_fm)
from fuzzyai.llm.providers.enums import LLMProvider, LLMProviderExtraParams
from fuzzyai.llm.providers.openai.models import OpenAIChatRequest
from fuzzyai.llm.providers.shared.decorators import api_endpoint, sync_api_endpoint

logger = logging.getLogger(__name__)

class OpenAIProviderException(BaseLLMProviderException):
    pass

class OpenAIConfig:
    API_BASE_URL = "https://api.openai.com/v1"
    CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"
    API_KEY_ENV_VAR = EnvironmentVariables.OPENAI_API_KEY.value
    O1_FAMILY_MODELS = {"o1-mini", "o1-preview", "o3-mini"}

@llm_provider_fm.flavor(LLMProvider.OPENAI)
class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str, **extra: Any):
        super().__init__(model=model, **extra)

        if (api_key := os.environ.get(OpenAIConfig.API_KEY_ENV_VAR)) is None:
            raise BaseLLMProviderException(f"{OpenAIConfig.API_KEY_ENV_VAR} not in os.environ")

        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        self._session = aiohttp.ClientSession(headers=self._headers)
        self._base_url = OpenAIConfig.API_BASE_URL

        try:
            self._tokenizer: Optional[tiktoken.Encoding] = tiktoken.encoding_for_model(model_name=model)
            self.tokens_handler = TokensHandler(tokenizer=self._tokenizer)
        except Exception as ex:
            logger.warning(f"Tokenizer not initialized: for model {model}, some attacks might not function properly")
            self.tokens_handler = None
            self._tokenizer = None

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "o1-mini", "o1-preview", "o3-mini", "gpt-4.5-preview"]

    
    @api_endpoint(OpenAIConfig.CHAT_COMPLETIONS_ENDPOINT)
    async def generate(self, prompt: str, url: str, system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        messages = [BaseLLMMessage(role=LLMRole.USER, content=prompt)]
        messages = self._prepare_messages(messages, system_prompt)
        return await self.chat(messages=messages, **extra) # type: ignore
    
    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    @api_endpoint(OpenAIConfig.CHAT_COMPLETIONS_ENDPOINT)
    async def chat(self, messages: list[BaseLLMMessage], url: str, system_prompt: Optional[str] = None, **extra: Any) -> BaseLLMProviderResponse:
        messages = self._prepare_messages(messages, system_prompt)
        try:
            request = OpenAIChatRequest(model=self._model_name, messages=messages, **extra)
            async with self._session.post(url, json=request.model_dump()) as response:
                openai_response = await response.json()

                self._handle_error_response(openai_response)
                choice = openai_response["choices"][0]
                if choice.get('finish_reason') == 'length':
                    logger.warning('OpenAI response was truncated! Please increase the token limit by setting -N=<max tokens>')

                return BaseLLMProviderResponse(response=choice['message']['content'])
        except (BaseLLMProviderRateLimitException, OpenAIProviderException) as e:
            raise e
        except Exception as e:            
            logger.error(f'Error generating text: {e}')
            raise OpenAIProviderException('Cant generate text')
    
    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        messages = [BaseLLMMessage(role=LLMRole.USER, content=prompt)]
        
        if extra.get(LLMProviderExtraParams.APPEND_LAST_RESPONSE) and (history := self.get_history()):
            messages.append(BaseLLMMessage(role=LLMRole.ASSISTANT, content=history[-1].response))
        
        chat_extra_params = {k:v for k, v in extra.items() if k not in [LLMProviderExtraParams.APPEND_LAST_RESPONSE]}
        return self.sync_chat(messages, **chat_extra_params)  # type: ignore

    @sync_api_endpoint(OpenAIConfig.CHAT_COMPLETIONS_ENDPOINT)
    def sync_chat(self, messages: list[BaseLLMMessage], url: str, 
                  system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        messages = self._prepare_messages(messages, system_prompt)

        try:
            request = OpenAIChatRequest(model=self._model_name, messages=messages, **extra)
            with requests.post(url, json=request.model_dump(), headers=self._headers) as response:
                openai_response = response.json()
                self._handle_error_response(openai_response)                    
                return BaseLLMProviderResponse(response=openai_response["choices"][0]['message']['content'])
        except (BaseLLMProviderRateLimitException, OpenAIProviderException) as e:
            raise e
        except Exception as e:            
            logger.error(f'Error generating text: {e}')
            raise OpenAIProviderException('Cant generate text')
    
    async def close(self) -> None:
        await self._session.close()

    def _prepare_messages(self, messages: list[BaseLLMMessage], 
                          system_prompt: Optional[str] = None) -> list[BaseLLMMessage]:
        if system_prompt and self._model_name not in OpenAIConfig.O1_FAMILY_MODELS:
            return [BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt)] + messages
        return messages
    
    @staticmethod
    def _handle_error_response(response_data: dict[str, Any]) -> None:
        if error := response_data.get("error"):
            if error.get("code") == "rate_limit_exceeded":
                raise BaseLLMProviderRateLimitException("Rate limit exceeded")
            raise OpenAIProviderException(f"OpenAI error: {error.get('message', 'Unknown error')}")

