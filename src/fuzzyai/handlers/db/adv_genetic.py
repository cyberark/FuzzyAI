from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from motor.motor_tornado import MotorClient
from pydantic import Field

from fuzzyai.consts import DATABASE_NAME, GENETIC_ATTACK_COLLECTION_NAME
from fuzzyai.db.mongodb import MongoDBHandler, MongoDocument


class AdversarialGeneticAttackDTO(MongoDocument):
    attack_id: str
    attack_start_timestamp: datetime
    attacked_model_name: str
    solution_index: int = Field(0)
    total_solutions_per_generation: int
    generation: int = Field(0)
    total_generations: int
    initial_suffix: str
    prompt_index: Optional[int] = Field(None)
    total_prompts: int

    prompt: str = Field(None)
    target: str = Field(None)
    suffix: str = Field(None)
    is_successful: bool = Field(False)
    loss: float = Field(None)
    model_response: str = Field(None)


class AdversarialGeneticAttackHandler(MongoDBHandler[AdversarialGeneticAttackDTO]):
    def __init__(
        self,
        client: MotorClient,
        database: str = DATABASE_NAME,
    ) -> None:
        super().__init__(
            client,
            database,
            GENETIC_ATTACK_COLLECTION_NAME,
            AdversarialGeneticAttackDTO,
        )
