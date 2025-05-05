from typing import Any

import motor.motor_asyncio

from fuzzyai.consts import DATABASE_NAME, FIELD_NAME_PROMPT, PROMPT_COLLECTION_NAME
from fuzzyai.db.mongodb import MongoDBHandler, MongoDocument
from fuzzyai.models.prompt import BasePrompt


class AdversarialPromptDTO(BasePrompt, MongoDocument):

    @classmethod
    def from_dict(cls: type['AdversarialPromptDTO'], **kwargs: Any) -> 'AdversarialPromptDTO':
        return cls(prompt=kwargs[FIELD_NAME_PROMPT])

    @classmethod
    def new(cls, *args: Any) -> 'AdversarialPromptDTO':
        return cls(prompt=args[0].strip())
    
    def __str__(self) -> str:
        return f"AdversarialPromptDTO(prompt={self.prompt})"


class AdversarialPromptsHandler(MongoDBHandler[AdversarialPromptDTO]):
    def __init__(self, client: motor.motor_asyncio.AsyncIOMotorClient, database: str = DATABASE_NAME) -> None:
        super().__init__(client, database, PROMPT_COLLECTION_NAME, AdversarialPromptDTO)
