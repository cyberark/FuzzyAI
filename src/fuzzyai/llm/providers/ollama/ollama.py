import asyncio
import logging
import random
from typing import Any, Optional, Union

import aiohttp

from fuzzyai.consts import OLLAMA_BASE_PORT
from fuzzyai.enums import LLMRole
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.llm.providers.ollama.models import (OllamaChatMessage, OllamaChatRequest, OllamaChatResponse,
                                                 OllamaGenerateRequest, OllamaGenerateResponse, OllamaOptions)
from fuzzyai.llm.providers.shared.decorators import api_endpoint

logger = logging.getLogger(__name__)

class OllamaProviderException(BaseLLMProviderException):
    pass


@llm_provider_fm.flavor(LLMProvider.OLLAMA)
class OllamaProvider(BaseLLMProvider):
    def __init__(self, llm_address: Optional[str] = None, 
                 ollama_port: int = OLLAMA_BASE_PORT, 
                 seed: int = 0, **extra: Any):
        super().__init__(**extra)
        
        if llm_address is None:
            llm_address = "localhost"
        
        self._base_url = f"http://{llm_address}:{int(ollama_port) + int(seed)}/api"
        
        random.seed(seed)
        self._seed = random.randint(42, 1024)
        self._session = aiohttp.ClientSession()
        self._validate_models_task = asyncio.create_task(self.validate_models())
    
    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return "<Model tag>"

    async def validate_models(self) -> None:
        async with self._session.get(f"{self._base_url}/tags") as response:
            models = await response.json()
            if not models:
                raise OllamaProviderException('No local ollama models found, Trying pulling some')
            
            model_names: list[str] = [model['name'] for model in models['models']]
            target_model = self._model_name if ":" in self._model_name else f"{self._model_name}:latest"
            if target_model not in model_names:
                raise OllamaProviderException(f"Model {self._model_name} not found in local ollama models, available models: {model_names}")
             
            
    @api_endpoint("/generate")
    async def generate(self, prompt: str, url: str, system_prompt: Optional[str] = None, raw: bool = False, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        await self._validate_models_task

        try:
            options = OllamaOptions(seed=self._seed, **extra)
            request = OllamaGenerateRequest(model=self._model_name, prompt=prompt, raw=raw,
                                            options=options)
            if system_prompt is not None:
                request.system = system_prompt

            async with self._session.post(url, json=request.model_dump()) as response:
                raw_response = await response.json()

                if 'error' in raw_response:
                    raise OllamaProviderException(f"Ollama error: {raw_response.get('error')}")
            
                ollama_response = OllamaGenerateResponse(**raw_response)
                return BaseLLMProviderResponse(response=ollama_response.response)
        except OllamaProviderException:
            raise
        except Exception as e:
            logger.error(f'Error generating text: {e}')
            raise OllamaProviderException('Cant generate text')

    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        return asyncio.run(self.generate(prompt, **extra)) # type: ignore
    
    @api_endpoint("/chat")
    async def chat(self, messages: list[BaseLLMMessage], url: str, system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        await self._validate_models_task

        try:
            options = OllamaOptions(seed=self._seed, **extra)
            ollama_messages = [OllamaChatMessage.model_validate(m.model_dump()) for m in messages]

            if system_prompt is not None:
                ollama_messages.insert(0, OllamaChatMessage(role=LLMRole.SYSTEM, content=system_prompt))

            request = OllamaChatRequest(model=self._model_name,
                                        messages=ollama_messages,
                                        options=options)
            
            async with self._session.post(url, json=request.model_dump()) as response:
                ollama_response = OllamaChatResponse(**(await response.json()))
                return BaseLLMProviderResponse(response=ollama_response.message.content)
        except Exception as e:
            logger.error(f'Error during chat: {e}')
            raise OllamaProviderException('Error during chat')
    
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError
        
    async def close(self) -> None:
        logger.debug(f"Closing OllamaProvider {self}")
        await self._session.close()
        try:
            self._validate_models_task.cancel()
        except:
            pass

    def __repr__(self) -> str:
        return f"model: {self._model_name}, base_url: {self._base_url}"
    
    def __str__(self) -> str:
        return f"model: {self._model_name}, base_url: {self._base_url}"
