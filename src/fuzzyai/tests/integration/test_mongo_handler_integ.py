from typing import Any

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from fuzzyai.handlers.db.adv_attacks import AdversarialAttackDTO, AdversarialAttacksHandler
from fuzzyai.handlers.db.adv_prompts import AdversarialPromptDTO, AdversarialPromptsHandler


@pytest.fixture
async def prompt_handler():
    database = "test_db"
    client = AsyncIOMotorClient("localhost", 27017)
    handler = AdversarialPromptsHandler(client, database=database)
    await client.drop_database(database)
    yield handler
    # Clean up after the test
    await client.drop_database(database)

@pytest.fixture
async def adv_attacks_handler():
    database = "test_db_adv_attacks"
    client = AsyncIOMotorClient("localhost", 27017)
    handler = AdversarialAttacksHandler(client, database=database)
    await client.drop_database(database)
    yield handler
    # Clean up after the test
    await client.drop_database(database)

async def test_retrieve_attack(adv_attacks_handler):
    attack1 = AdversarialAttackDTO(attack_id="123", attack_time="2021-01-01T00:00:00", prompt="hello there", suffix="!", response="hello there!", llm_type="gpt2", label="positive", sentiment_analysis={})
    attack2 = AdversarialAttackDTO(attack_id="456", attack_time="2021-01-01T00:00:00", prompt="hello there", suffix="!", response="hello there!", llm_type="gpt2", label="positive", sentiment_analysis={})
    await adv_attacks_handler.store([attack1, attack2])
    retrieved_attacks = await adv_attacks_handler.retrieve_by_property(key="attack_id", value="456")
    assert retrieved_attacks == [attack2]
    
async def test_save_prompts(prompt_handler):
    prompts = [AdversarialPromptDTO(prompt="hello there1"), AdversarialPromptDTO(prompt="hello there2"), AdversarialPromptDTO(prompt="hello there3")]
    await prompt_handler.store(prompts)
    assert await prompt_handler._collection.count_documents({}) == len(prompts)

async def test_retrieve_prompts(prompt_handler):
    prompts = [AdversarialPromptDTO(prompt="hello there1"), AdversarialPromptDTO(prompt="hello there2"), AdversarialPromptDTO(prompt="hello there3")]
    await prompt_handler.store(prompts)
    retrieved_prompts = await prompt_handler.retrieve()
    assert retrieved_prompts == prompts

async def test_retrieve_prompts_paging(prompt_handler):
    prompts = [AdversarialPromptDTO(prompt=f"hello there{n}") for n in range(2500)]
    await prompt_handler.store(prompts)
    result: list[dict[str, Any]] = [] # type: ignore
    result += await prompt_handler.retrieve()
    assert len(result) == 1000
    result += await prompt_handler.retrieve()
    assert len(result) == 2000
    result += await prompt_handler.retrieve()
    assert len(result) == 2500
    result += await prompt_handler.retrieve()
    assert len(result) == 2500
    assert result == prompts

async def test_retrieve_prompts_all(prompt_handler):
    prompts = [AdversarialPromptDTO(prompt=f"hello there{n}") for n in range(1001)]
    await prompt_handler.store(prompts)
    result: list[dict[str, Any]] = [] # type: ignore
    result += await prompt_handler.retrieve_all()
    assert len(result) == 1001
    assert result == prompts