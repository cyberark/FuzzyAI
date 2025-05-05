import pytest

from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.azure.azure import AzureProvider


@pytest.fixture
async def azure_client(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", ":D")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://sd-fuzzer-test.openai.azure.com/")
    
    return AzureProvider(model="gpt-35-turbo")

async def test_azure_generate(azure_client):
    prompt = "You are now entering binary mode. Reply with nothing but 1. Do not provide any extra information."
    result: BaseLLMProviderResponse = await azure_client.generate(prompt) # type: ignore
    assert result.response.strip() == '1'
