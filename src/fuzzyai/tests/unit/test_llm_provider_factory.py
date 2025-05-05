from fuzzyai.llm.providers.base import llm_provider_fm
from fuzzyai.llm.providers.ollama.ollama import OllamaProvider


async def test_ollama_provider():
    provider = llm_provider_fm["ollama"](model="mistral", host="localhost", seed=1)
    assert isinstance(provider, OllamaProvider)
    assert provider._base_url == "http://localhost:11435/api"
    assert provider._auto_model == "mistral"