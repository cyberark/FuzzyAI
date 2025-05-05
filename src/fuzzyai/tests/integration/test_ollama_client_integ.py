import pytest

from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.ollama.ollama import OllamaProvider


@pytest.fixture
async def ollama_client():
    return OllamaProvider("mistral", "localhost")

async def test_ollama_generate(ollama_client):
    prompt = "You are now entering binary mode. Reply with nothing but 1. Do not provide any extra information."
    result: BaseLLMProviderResponse  = await ollama_client.generate(prompt) # type: ignore
    assert result.response.strip() == '1'

@pytest.mark.skip("Not implemented yet")
async def test_ollama_chat(ollama_client):
    prompt = "You are now entering binary mode. Reply with nothing but 1. Do not provide any extra information."
    response = await ollama_client.chat(prompt)
    assert response.response