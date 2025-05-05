from typing import Any

import motor.motor_asyncio

from fuzzyai.consts import CLASSIFICATIONS_COLLECTION_NAME, DATABASE_NAME
from fuzzyai.db.mongodb import MongoDBHandler, MongoDocument


class AdversarialClassificationDTO(MongoDocument):
    execution_id: str
    classifications: list[dict[str, Any]]

    def __str__(self) -> str:
        return f"AdversarialClassificationDTO(classifications={self.classifications})"


class AdversarialClassificationHandler(MongoDBHandler[AdversarialClassificationDTO]):
    def __init__(self, client: motor.motor_asyncio.AsyncIOMotorClient, database: str = DATABASE_NAME) -> None:
        super().__init__(client, database, CLASSIFICATIONS_COLLECTION_NAME, AdversarialClassificationDTO)
