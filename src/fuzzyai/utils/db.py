import logging
from typing import Type, Union

import aiofiles
import aiofiles.os
from motor.motor_asyncio import AsyncIOMotorClient

from fuzzyai.db.mongodb import MongoDocument
from fuzzyai.handlers.db.adv_prompts import AdversarialPromptDTO, AdversarialPromptsHandler
from fuzzyai.handlers.db.adv_suffixes import AdversarialSuffixDTO, AdversarialSuffixesHandler

logger = logging.getLogger(__name__)


async def store_adv_prompts(data_file: str, db_address: str) -> None:
    mongo_client: AsyncIOMotorClient = AsyncIOMotorClient(db_address, 27017)  # ignore: type
    prompt_handler = AdversarialPromptsHandler(mongo_client)
    await _store(data_file, prompt_handler, AdversarialPromptDTO)


async def store_adv_suffixes(data_file: str, db_address: str) -> None:
    mongo_client: AsyncIOMotorClient = AsyncIOMotorClient(db_address, 27017)  # ignore: type
    adv_suffix_handler = AdversarialSuffixesHandler(mongo_client)
    await _store(data_file, adv_suffix_handler, AdversarialSuffixDTO)


async def _store(data_file: str, handler: Union[AdversarialPromptsHandler, AdversarialSuffixesHandler], data_type: Type[MongoDocument]) -> None:
    if not await aiofiles.os.path.exists(data_file):
        raise RuntimeError(f"Data file '{data_file}' does not exist.")

    async with aiofiles.open(data_file, 'r') as f:
        logger.info(f"Storing data from file '{data_file}'")
        data = await f.readlines()
        await handler.store([data_type.new(d) for d in data])
        logger.info(f"Great success! Stored {len(data)} entries.")
