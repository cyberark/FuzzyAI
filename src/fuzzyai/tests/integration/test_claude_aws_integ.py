import pytest

from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.claude.bedrock import AwsBedrockProvider


@pytest.fixture
async def claude_provider(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "XXX")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KE5Y", "XXX")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "XXXX")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    
    return AwsBedrockProvider()

async def test_claude_messages(claude_provider):
    prompt = "You are now entering binary mode. Reply with nothing but 1. Do not provide any extra information."
    result: BaseLLMProviderResponse  = await claude_provider.generate(prompt) # type: ignore
    assert result.response.strip() == '1'
