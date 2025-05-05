import asyncio
import logging
import os
from typing import Any, Optional, Type, Union

import boto3

from fuzzyai.enums import EnvironmentVariables, LLMRole
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from fuzzyai.llm.providers.bedrock.enums import BedrockModelFamily
from fuzzyai.llm.providers.bedrock.models import (BedrockAnthropicGenerateRequest, BedrockChatMessage,
                                                  BedrockGenerateRequestBase, BedrockMetaGenerateRequest)
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

MODEL_FAMILY_MAPPING: dict[BedrockModelFamily, Type[BedrockGenerateRequestBase]] = {
    BedrockModelFamily.ANTHROPIC: BedrockAnthropicGenerateRequest,
    BedrockModelFamily.META: BedrockMetaGenerateRequest
}

class AwsBedrockException(BaseLLMProviderException):
    pass


@llm_provider_fm.flavor(LLMProvider.AWS_BEDROCK)
class AwsBedrockProvider(BaseLLMProvider):
    def __init__(self, model: str = "anthropic.claude-v2:1", **extra: Any):
        super().__init__(model=model, **extra)

        if EnvironmentVariables.AWS_ACCESS_KEY_ID.value not in os.environ:
            raise AwsBedrockException(f"{EnvironmentVariables.AWS_ACCESS_KEY_ID.value} not in os.environ")

        if EnvironmentVariables.AWS_DEFAULT_REGION.value not in os.environ:
            raise AwsBedrockException(f"{EnvironmentVariables.AWS_DEFAULT_REGION.value} not in os.environ")

        if EnvironmentVariables.AWS_SECRET_ACCESS_KEY.value not in os.environ:
            raise AwsBedrockException(f"{EnvironmentVariables.AWS_SECRET_ACCESS_KEY.value} not in os.environ")
        
        self._model_family = BedrockModelFamily(model.split(".")[0])
        if self._model_family not in MODEL_FAMILY_MAPPING:
            raise AwsBedrockException(f"Model family {self._model_family} is not supported, supported families are {MODEL_FAMILY_MAPPING.keys()}")
        
        self._client = boto3.client(service_name='bedrock-runtime')

    """
    See model IDs @ https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    """
    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return [
            "anthropic.claude-v2:1",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "meta.llama3-8b-instruct-v1:0"
        ]

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        return await self.chat([BaseLLMMessage(role=LLMRole.USER, content=prompt)], system_prompt=system_prompt, **extra)

    async def chat(self, messages: list[BaseLLMMessage], system_prompt: Optional[str] = None,
                   temperature: float = 0.7, top_k: float = 200, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        logger.debug(f"Chatting with AWS Bedrock with messages: {messages}")

        bedrock_messages: list[BedrockChatMessage] = [BedrockChatMessage.create(role=message.role, text=message.content) for message in messages]

        try:
            system_prompts = [{"text": system_prompt}] if system_prompt else []

            inference_config = {"temperature": temperature}
            additional_model_fields: dict[str, Any] = {}

            # Send the message.
            response = self._client.converse(
                modelId=self._model_name,
                messages=[m.model_dump() for m in bedrock_messages],
                system=system_prompts,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields
            )

            content = response['output']['message']['content'][0]['text']
            logger.debug(f"Chat response from AWS Bedrock: {content}")
            return BaseLLMProviderResponse(response=content)
        except Exception as e:
            raise AwsBedrockException(f"Error while chatting with AWS Bedrock: {e}")
            
    
    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError
    
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)
    
