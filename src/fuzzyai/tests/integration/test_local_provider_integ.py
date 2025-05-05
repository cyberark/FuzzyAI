import pytest

from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.local.local_llm import LocalProvider


@pytest.fixture
async def local_provider():
    return LocalProvider(model="/Users/shaid/Desktop/research/models/Llama-2-7b-chat-hf")

async def test_local_generate(local_provider):
    prompt = "You are now entering binary mode. Reply with nothing but 1. Do not provide any extra information. Lets start: send a reply"
    result: BaseLLMProviderResponse  = await local_provider.generate(prompt) # type: ignore
    assert result.response.strip() == '1'