import logging
import os
from typing import Any, Optional, Union

import aiohttp
import backoff

from fuzzyai.enums import EnvironmentVariables, LLMRole
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import (BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException,
                                        BaseLLMProviderRateLimitException, llm_provider_fm)
from fuzzyai.llm.providers.deepseek.models import DeepSeekChatRequest
from fuzzyai.llm.providers.enums import LLMProvider, LLMProviderExtraParams
from fuzzyai.llm.providers.shared.decorators import api_endpoint

DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"

logger = logging.getLogger(__name__)

class DeepSeekProviderException(BaseLLMProviderException):
    pass


@llm_provider_fm.flavor(LLMProvider.DEEPSEEK)
class DeepSeekProvider(BaseLLMProvider):
    CHAT_COMPLETIONS_URL = f"{DEEPSEEK_API_BASE_URL}/chat/completions"

    def __init__(self, model: str, **extra: Any):
        super().__init__(model=model, **extra)

        if (api_key := os.environ.get(EnvironmentVariables.DEEPSEEK_API_KEY.value)) is None:
            raise BaseLLMProviderException(f"{EnvironmentVariables.DEEPSEEK_API_KEY.value} not in os.environ")

        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        self._session = aiohttp.ClientSession(headers=self._headers)
        self._base_url = DEEPSEEK_API_BASE_URL

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return ["deepseek-chat", "deepseek-reasoner"]

    @api_endpoint("/chat/completions")
    async def generate(self, prompt: str, url: str, system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        messages = [BaseLLMMessage(role=LLMRole.USER, content=prompt)]
        if system_prompt is not None:
            messages = [BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt)] + messages

        return await self.chat(messages=messages, **extra) # type: ignore

    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    @api_endpoint("/chat/completions")
    async def chat(self, messages: list[BaseLLMMessage], url: str, system_prompt: Optional[str] = None, **extra: Any) -> BaseLLMProviderResponse:
        error: dict[str, Any]
        
        if system_prompt is not None:
            messages = [BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt)] + messages

        try:
            request = DeepSeekChatRequest(model=self._model_name, messages=messages, **extra)
            async with self._session.post(url, json=request.model_dump()) as response:
                deepseek_response = await response.json()
                if (error := deepseek_response.get("error")) is not None:
                    if error.get('code') == 'rate_limit_exceeded':
                        logger.debug(f'Rate limit exceeded')
                        raise BaseLLMProviderRateLimitException()
                    else:
                        raise DeepSeekProviderException('DeepSeek error: ' + error.get('message'))
                    
                return BaseLLMProviderResponse(response=deepseek_response["choices"][0]['message']['content'])
        except (BaseLLMProviderRateLimitException, DeepSeekProviderException) as e:
            raise e
        except Exception as e:            
            logger.error(f'Error generating text: {e}')
            raise DeepSeekProviderException('Cant generate text')
    
    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        messages = [BaseLLMMessage(role=LLMRole.USER, content=prompt)]
        
        if extra.get(LLMProviderExtraParams.APPEND_LAST_RESPONSE):
            if history := self.get_history():
                messages.append(BaseLLMMessage(role=LLMRole.ASSISTANT, content=history[-1].response))
        
        chat_extra_params = {k:v for k, v in extra.items() if k not in [LLMProviderExtraParams.APPEND_LAST_RESPONSE]}
        return self.sync_chat(messages, **chat_extra_params)

    async def close(self) -> None:
        await self._session.close()
