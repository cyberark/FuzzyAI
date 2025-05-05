import logging
import os
from typing import Any, Optional, Union

import aiohttp

from fuzzyai.enums import EnvironmentVariables, LLMRole
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.azure.models import AzureGenerateOptions, AzureMessage, AzureRequest
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

class AzureProviderException(BaseLLMProviderException):
    pass

AZURE_ENDPOINT_URL = "{endpoint}/openai/deployments/{model}/chat/completions?api-version={version}"

@llm_provider_fm.flavor(LLMProvider.AZURE)
class AzureProvider(BaseLLMProvider):
    def __init__(self, model: str, **extra: Any):
        super().__init__(model=model, **extra)

        if (api_key := os.environ.get(EnvironmentVariables.AZURE_OPENAI_API_KEY.value)) is None:
            raise BaseLLMProviderException(f"{EnvironmentVariables.AZURE_OPENAI_API_KEY.value} not in os.environ")
        
        if (endpoint := os.environ.get(EnvironmentVariables.AZURE_OPENAI_ENDPOINT.value)) is None:
            raise BaseLLMProviderException(f"{EnvironmentVariables.AZURE_OPENAI_ENDPOINT.value} not in os.environ")
        
        self._base_url = AZURE_ENDPOINT_URL.format(endpoint=endpoint, 
                                                   model=self._model_name, 
                                                   version="2023-07-01-preview")
        
        self._session = aiohttp.ClientSession(headers={
            "Content-Type": "application/json",
            "api-key": api_key
        })

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return ["gpt-35-turbo", "gpt-4", "gpt-4o"]
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            options = AzureGenerateOptions.model_validate(extra)
            request = AzureRequest(messages=[AzureMessage(content=prompt)], **options.model_dump())
            if system_prompt:
                request.messages.insert(0, AzureMessage(role=LLMRole.SYSTEM, content=system_prompt))

            async with self._session.post(self._base_url, json=request.model_dump()) as response:
                azure_response = await response.json()
                if 'error' in azure_response:
                    raise AzureProviderException(azure_response['error']['message'])

                return BaseLLMProviderResponse(response=azure_response["choices"][0]['message']['content'])
        except AzureProviderException as e:
            raise e
        except Exception as e:
            logger.error(f'Error generating text: {e}')
            raise AzureProviderException('Cant generate text')
    
    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError
    
    async def chat(self, messages: list[BaseLLMMessage], **extra: Any) -> BaseLLMProviderResponse:
        raise NotImplementedError

    async def close(self) -> None:
        await self._session.close()
