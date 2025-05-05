
from typing import Any

import motor.motor_asyncio

from fuzzyai.consts import DATABASE_NAME, TAXONOMIES_COLLECTION_NAME
from fuzzyai.db.mongodb import MongoDBHandler, MongoDocument
from fuzzyai.models.prompt import BasePrompt


class AdversarialTaxonomyDTO(BasePrompt, MongoDocument):
    execution_id: str
    responses: dict[str, Any]
    
    def __str__(self) -> str:
        return f"AdversarialPromptDTO(prompt={self.prompt}, responses={self.responses})"


class AdversarialTaxonomyHandler(MongoDBHandler[AdversarialTaxonomyDTO]):
    def __init__(self, client: motor.motor_asyncio.AsyncIOMotorClient, database: str = DATABASE_NAME) -> None:
        super().__init__(client, database, TAXONOMIES_COLLECTION_NAME, AdversarialTaxonomyDTO)
