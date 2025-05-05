import logging
import os
from typing import Any, Optional, Union

import aiohttp
import backoff
import requests

from fuzzyai.enums import EnvironmentVariables, LLMRole
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.ai21.models import AI21ChatRequest
from fuzzyai.llm.providers.base import (BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException,
                                        BaseLLMProviderRateLimitException, llm_provider_fm)
from fuzzyai.llm.providers.enums import LLMProvider, LLMProviderExtraParams
from fuzzyai.llm.providers.shared.decorators import api_endpoint

logger = logging.getLogger(__name__)


class AI21ProviderException(BaseLLMProviderException):
    pass

AI21_API_BASE_URL = "https://api.ai21.com/studio/v1"

@llm_provider_fm.flavor(LLMProvider.AI21)
class AI21Provider(BaseLLMProvider):
    CHAT_COMPLETIONS_URL = f"{AI21_API_BASE_URL}/chat/completions"

    def __init__(self, model: str, **extra: Any):
        super().__init__(model=model, **extra)

        if (api_key := os.environ.get(EnvironmentVariables.AI21_API_KEY.value)) is None:
            raise BaseLLMProviderException(f"{EnvironmentVariables.AI21_API_KEY.value} not in os.environ")

        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self._session = aiohttp.ClientSession(headers=self._headers)

        self._base_url = AI21_API_BASE_URL        

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return [
            "jamba-1.5-mini",
            "jamba-1.5-large",
        ]

    @api_endpoint("/chat/completions")
    async def generate(
        self, prompt: str, url: str, system_prompt: Optional[str] = None, **extra: Any
    ) -> Optional[BaseLLMProviderResponse]:
        messages = [BaseLLMMessage(role=LLMRole.USER, content=prompt)]
        if system_prompt is not None:
            messages = [
                BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt)
            ] + messages

        return await self.chat(messages=messages, **extra)  # type: ignore

    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    @api_endpoint("/chat/completions")
    async def chat(
        self,
        messages: list[BaseLLMMessage],
        url: str,
        system_prompt: Optional[str] = None,
        **extra: Any,
    ) -> BaseLLMProviderResponse:
        try:
            if system_prompt is not None:
                messages = [
                    BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt)
                ] + messages

            request = AI21ChatRequest(
                model=self._model_name, messages=messages, **extra
            )

            async with self._session.post(url, json=request.model_dump()) as response:
                ai21_response = await response.json()
                if (error := ai21_response.get("error")) is not None:
                    if error.get('code') == 'rate_limit_exceeded':
                        logger.debug(f'Rate limit exceeded')
                        raise BaseLLMProviderRateLimitException()
                    else:
                        raise AI21ProviderException(f"AI21 error: {error.get('message')}")
                    
                return BaseLLMProviderResponse(
                    response=ai21_response["choices"][0]['message']['content']
                )
        except (BaseLLMProviderRateLimitException, AI21ProviderException) as e:
            raise e
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise AI21ProviderException("Cant generate text")

    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    def sync_generate(
        self, prompt: str, **extra: Any
    ) -> Optional[BaseLLMProviderResponse]:
        messages = [BaseLLMMessage(role=LLMRole.USER, content=prompt)]

        if extra.get(LLMProviderExtraParams.APPEND_LAST_RESPONSE):
            if history := self.get_history():
                messages.append(
                    BaseLLMMessage(role=LLMRole.ASSISTANT, content=history[-1].response)
                )

        chat_extra_params = {
            k: v
            for k, v in extra.items()
            if k not in [LLMProviderExtraParams.APPEND_LAST_RESPONSE]
        }
        return self.sync_chat(messages, **chat_extra_params)

    def sync_chat(
        self, messages: list[BaseLLMMessage], **extra: Any
    ) -> BaseLLMProviderResponse:
        try:
            request = AI21ChatRequest(
                model=self._model_name, messages=messages, **extra
            )
            response = requests.post(self.CHAT_COMPLETIONS_URL, headers=self._headers, json=request.model_dump())
            ai21_response = response.json()
            if (error := ai21_response.get("error")) is not None:
                if error.get('code') == 'rate_limit_exceeded':
                    logger.debug(f'Rate limit exceeded')
                    raise BaseLLMProviderRateLimitException()
                else:
                    raise AI21ProviderException(f"AI21 error: {error.get('message')}")

            return BaseLLMProviderResponse(
                response=ai21_response["choices"][0]["message"]["content"]
            )
        except (BaseLLMProviderRateLimitException, AI21ProviderException) as e:
            raise e
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise AI21ProviderException("Cant generate text")

    async def close(self) -> None:
        await self._session.close()
