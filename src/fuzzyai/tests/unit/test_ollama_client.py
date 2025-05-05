import dataclasses
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from fuzzyai.llm.providers.ollama.ollama import OllamaChatResponse, OllamaGenerateResponse, OllamaProvider


@dataclasses.dataclass
class MockPack:
    llm_generate_mock: AsyncMock
    llm_chat_mock: AsyncMock

@pytest.fixture
async def ollama_client():
    return OllamaProvider(model="mistral", host="localhost", port=8080)

@pytest.fixture
def llm_generate_mock() -> AsyncMock:
    default_response = OllamaGenerateResponse(response="Test response")
    return AsyncMock(return_value=web.json_response(default_response.model_dump()))

@pytest.fixture
def llm_chat_mock() -> AsyncMock:
    default_response = OllamaChatResponse(done=True)
    return AsyncMock(return_value=web.json_response(default_response.model_dump()))

@pytest.fixture
def mock_pack(llm_generate_mock: AsyncMock, llm_chat_mock: AsyncMock) -> MockPack:
    return MockPack(llm_chat_mock=llm_chat_mock, llm_generate_mock=llm_generate_mock)

@pytest.fixture
async def server(unused_tcp_port: int, mock_pack: MockPack) -> web.TCPSite:
    routes = web.RouteTableDef()

    @routes.post('/api/generate')
    async def generate_mock(request: web.Request) -> web.Response:
        payload = await request.json()
        return await mock_pack.llm_generate_mock(payload)

    @routes.post('/api/chat')
    async def chat_mock(request: web.Request) -> web.Response:
        payload = await request.json()
        return await mock_pack.llm_chat_mock(payload)
    
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    yield site

    await site.stop()
    await runner.cleanup()

async def test_generate(server: web.TCPSite, mock_pack: MockPack):
    prompt = "Test prompt"

    client = OllamaProvider(model="mistral", llm_address="localhost", seed=0, port=server._port, system_prompt="Test system prompt")
    result: OllamaGenerateResponse = await client.generate(prompt) # type: ignore
    assert result.response == "Test response"
    mock_pack.llm_generate_mock.assert_awaited_once()

async def test_chat(server: web.TCPSite, mock_pack: MockPack):
    prompt = "Test prompt"

    client = OllamaProvider(model="mistral", llm_address="localhost", seed=0, port=server._port)
    result: OllamaChatResponse = await client.chat(prompt) # type: ignore
    assert result
    mock_pack.llm_chat_mock.assert_awaited_once()

