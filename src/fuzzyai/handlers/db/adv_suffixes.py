from typing import Any

import motor.motor_asyncio

from fuzzyai.consts import ADVERSARIAL_SUFFIX_COLLECTION_NAME, DATABASE_NAME, FIELD_NAME_SUFFIX
from fuzzyai.db.mongodb import MongoDBHandler, MongoDocument


class AdversarialSuffixDTO(MongoDocument):
    suffix: str

    @classmethod
    def from_dict(cls, **kwargs: Any) -> 'AdversarialSuffixDTO':
        return cls(suffix=kwargs[FIELD_NAME_SUFFIX])

    @classmethod
    def new(cls, *args: Any) -> 'AdversarialSuffixDTO':
        return cls(suffix=args[0].strip())


class AdversarialSuffixesHandler(MongoDBHandler[AdversarialSuffixDTO]):
    def __init__(self, client: motor.motor_asyncio.AsyncIOMotorClient,
                 database: str = DATABASE_NAME) -> None:  # ignore: type
        super().__init__(client, database, ADVERSARIAL_SUFFIX_COLLECTION_NAME, AdversarialSuffixDTO)
