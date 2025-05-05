import logging

import motor.motor_asyncio

from fuzzyai.consts import ADVERSARIAL_ATTACK_COLLECTION_NAME, DATABASE_NAME
from fuzzyai.db.mongodb import MongoDBHandler
from fuzzyai.handlers.attacks.models import AttackSummary

logger = logging.getLogger(__name__)

class AdversarialAttacksHandler(MongoDBHandler[AttackSummary]):
    def __init__(self, client: motor.motor_asyncio.AsyncIOMotorClient, database: str = DATABASE_NAME) -> None:
        super().__init__(client, database, ADVERSARIAL_ATTACK_COLLECTION_NAME, AttackSummary)
