from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import motor.motor_asyncio
from pydantic import Field

from fuzzyai.consts import DATABASE_NAME, GCG_ATTACK_COLLECTION_NAME
from fuzzyai.db.mongodb import MongoDBHandler, MongoDocument


class AdversarialGCGAttackDTO(MongoDocument):
    attack_id: str
    attack_start_timestamp: datetime
    attacked_model_name: str
    step: int = Field(0)
    steps: int
    prompt_index: int = Field(None)
    total_prompts: int
    initial_suffix: str

    prompt: str = Field(None)
    target: str = Field(None)
    suffix: str = Field(None)
    is_successful: bool = Field(False)
    loss: float = Field(None)
    model_response: str = Field(None)
    
    # necessary because the gcg attack handler uses old pydantic version which doesn't include this method by default
    def model_dump(self) -> dict[str, Any]: # type: ignore
        return self.dict()


class AdversarialGCGAttackHandler(MongoDBHandler[AdversarialGCGAttackDTO]):
    def __init__(
        self,
        client: motor.motor_asyncio.AsyncIOMotorClient,
        database: str = DATABASE_NAME,
    ) -> None:
        super().__init__(client, database, GCG_ATTACK_COLLECTION_NAME, AdversarialGCGAttackDTO)
