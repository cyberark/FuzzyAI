import dataclasses
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from fuzzyai.llm.providers.azure.azure import AzureProvider


@dataclasses.dataclass
class MockPack:
    llm_completion_mock: AsyncMock

@pytest.fixture
def llm_completion_mock() -> AsyncMock:
    default_response = {
        "choices": [
            {
                "finishReason": "length",
                "index": 0,
                "logprobs": None,
                "text": "1",
                "tokenLogprobs": None,
                "message": {
                    "role": "assistant",
                    "content": "test response"
                }
            }
        ],
    }

    return AsyncMock(return_value=web.json_response(default_response))

@pytest.fixture
def mock_pack(llm_completion_mock: AsyncMock) -> MockPack:
    return MockPack(llm_completion_mock=llm_completion_mock)

@pytest.fixture
async def server(unused_tcp_port: int, mock_pack: MockPack) -> web.TCPSite:
    routes = web.RouteTableDef()

    @routes.post('/openai/deployments/gpt-35-turbo/chat/completions')
    async def completions_mock(request: web.Request) -> web.Response:
        payload = await request.json()
        return await mock_pack.llm_completion_mock(payload)
    
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    yield site

    await site.stop()
    await runner.cleanup()

async def test_generate(monkeypatch, server: web.TCPSite, mock_pack: MockPack):
    prompt = "Test prompt"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", ":D")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", f"http://127.0.0.1:{server._port}")
    
    client = AzureProvider(model="gpt-35-turbo")
    result = await client.generate(prompt,  max_tokens=10) # type: ignore
    assert result.response == "test response"

    mock_pack.llm_completion_mock.assert_awaited_once()
