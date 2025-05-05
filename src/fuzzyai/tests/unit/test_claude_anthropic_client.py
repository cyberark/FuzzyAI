import dataclasses
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from fuzzyai.llm.providers.anthropic.handler import AnthropicProvider
from fuzzyai.llm.providers.anthropic.models import AnthropicMessage, AnthropicMessagesResponse
from fuzzyai.llm.providers.base import BaseLLMProviderResponse


@dataclasses.dataclass
class MockPack:
    llm_messages_mock: AsyncMock

@pytest.fixture
def llm_messages_mock() -> AsyncMock:
    default_response = AnthropicMessagesResponse(type="message", messages=[
        AnthropicMessage(type="message", text="Test response")
    ])
    return AsyncMock(return_value=web.json_response(default_response.model_dump()))

@pytest.fixture
def mock_pack(llm_messages_mock: AsyncMock) -> MockPack:
    return MockPack(llm_messages_mock=llm_messages_mock)

@pytest.fixture
async def server(unused_tcp_port: int, mock_pack: MockPack) -> web.TCPSite:
    routes = web.RouteTableDef()

    @routes.post('/messages')
    async def messages_mock(request: web.Request) -> web.Response:
        payload = await request.json()
        return await mock_pack.llm_messages_mock(payload)
    
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    yield site

    await site.stop()
    await runner.cleanup()

async def test_messages(server: web.TCPSite, mock_pack: MockPack, monkeypatch):
    monkeypatch.setenv(AnthropicProvider.ANTHROPIC_API_KEY, "test")

    prompt = "Test prompt"

    client = AnthropicProvider(url_override=f"http://localhost:{server._port}")
    result: BaseLLMProviderResponse = await client.generate(prompt) # type: ignore
    assert result.response == "Test response"
    mock_pack.llm_messages_mock.assert_awaited_once()
