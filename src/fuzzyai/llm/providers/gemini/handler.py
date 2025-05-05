import logging
import os
from typing import Any, Optional, Union

import aiohttp
import backoff

from fuzzyai.enums import EnvironmentVariables, LLMRole
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import (BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException,
                                        BaseLLMProviderRateLimitException, llm_provider_fm)
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.llm.providers.gemini.models import GenerateContentRequest, GenerateContentResponse, SafetySetting
from fuzzyai.llm.providers.shared.decorators import api_endpoint

logger = logging.getLogger(__name__)

class GeminiProviderException(BaseLLMProviderException):
    pass

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1/models/"

@llm_provider_fm.flavor(LLMProvider.GEMINI)
class GeminiProvider(BaseLLMProvider):
    def __init__(self, model: str, safety_settings: Optional[list[SafetySetting]] = None, **extra: Any):
        super().__init__(model=model, **extra)

        if (api_key := os.environ.get(EnvironmentVariables.GEMINI_API_KEY.value)) is None:
            raise GeminiProviderException(f"{EnvironmentVariables.GEMINI_API_KEY.value} not in os.environ")

        self._session = aiohttp.ClientSession(headers={
            "Content-Type": "application/json",
        })

        self._api_key = api_key
        self._safety_settings = safety_settings or []
        self._base_url = GEMINI_API_BASE_URL + self._model_name + ":"

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    
    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    @api_endpoint("generateContent")
    async def generate(self, prompt: str, url: str, system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        messages = [BaseLLMMessage(role=LLMRole.USER, content=prompt)]
        if system_prompt is not None:
            messages.insert(0, BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt))

        return await self.chat([BaseLLMMessage(role=LLMRole.USER, content=prompt)], **extra) # type: ignore
    
    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    @api_endpoint("generateContent")
    async def chat(self, messages: list[BaseLLMMessage], url: str, system_prompt: Optional[str] = None, **extra: Any) -> BaseLLMProviderResponse:
        try:
            if system_prompt is not None and not any(m.role == LLMRole.SYSTEM for m in messages):
                messages.insert(0, BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt))

            request = GenerateContentRequest.from_messages(messages, **extra)
            if self._safety_settings:
                request.safety_settings = self._safety_settings

            async with self._session.post(url, data=request.model_dump_json(by_alias=True), params={"key": self._api_key}) as r:
                gemini_response = await r.json()
                if r.status != 200:
                    raise GeminiProviderException(f"Error generating text: {gemini_response}")
                
                response: GenerateContentResponse = GenerateContentResponse.model_validate(gemini_response)
                return BaseLLMProviderResponse(response=response.get_content() or str())
        except (GeminiProviderException, BaseLLMProviderRateLimitException) as e:
            raise e
        except Exception as e:            
            logger.error(f'Error generating text: {e}')
            raise GeminiProviderException('Cant generate text')
    
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    async def close(self) -> None:
        await self._session.close()
