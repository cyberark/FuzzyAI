import logging
import os
from typing import Any, Optional, Union

import aiohttp
import backoff

from fuzzyai.enums import EnvironmentVariables
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.anthropic.models import (AnthropicGenerateOptions, AnthropicMessagesRequest,
                                                    AnthropicMessagesResponse)
from fuzzyai.llm.providers.base import (BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException,
                                        BaseLLMProviderRateLimitException, llm_provider_fm)
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.llm.providers.shared.decorators import api_endpoint

logger = logging.getLogger(__name__)

class AnthropicProviderException(BaseLLMProviderException):
    pass


@llm_provider_fm.flavor(LLMProvider.ANTHROPIC)
class AnthropicProvider(BaseLLMProvider):
    def __init__(self, model: str = "claude-2.1", url_override: Optional[str] = None, 
                 **extra: Any):
        super().__init__(model=model, **extra)

        if (api_key := os.environ.get(EnvironmentVariables.ANTHROPIC_API_KEY.value)) is None:
            raise BaseLLMProviderException(f"{EnvironmentVariables.ANTHROPIC_API_KEY.value} not in os.environ")
        
        self._session = aiohttp.ClientSession(headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15"
        })

        self._base_url = url_override or "https://api.anthropic.com/v1"

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return ["claude-2.1", "claude-3-haiku-20240307", "claude-3-opus-latest", "claude-3-sonnet-20240229", "claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-3-7-sonnet-20250219"]

    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    @api_endpoint("/messages")
    async def generate(self, prompt: str, url: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        messages = [BaseLLMMessage(role="user", content=prompt)]
        return await self.chat(messages, **extra) or None
    
    @api_endpoint("/messages")
    async def chat(self, messages: list[BaseLLMMessage], url: str, system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            options = AnthropicGenerateOptions.model_validate(extra)
            request = AnthropicMessagesRequest(model=self._model_name, system=system_prompt, messages=messages, **options.model_dump())

            async with self._session.post(url, json=request.model_dump()) as response:
                response_json: dict[str, Any] = await response.json()
                error: Optional[dict[str, str]]
                
                if (error := response_json.get("error")) is not None:
                    if error.get('message') == 'Overloaded':
                        raise BaseLLMProviderRateLimitException()
                    raise AnthropicProviderException(f"Anthropic error: {error.get('message')}")
                
                result: AnthropicMessagesResponse = AnthropicMessagesResponse(**(await response.json()))
                if result.content:
                    return BaseLLMProviderResponse(response=result.content[0].text)
                
                return None
        except (BaseLLMProviderRateLimitException, AnthropicProviderException):
            raise
        except Exception as e:
            logger.error(f'Error generating text: {e}')
            raise AnthropicProviderException('Cant generate text')
        
    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError
    
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError
    
    async def close(self) -> None:
        await self._session.close()
    
