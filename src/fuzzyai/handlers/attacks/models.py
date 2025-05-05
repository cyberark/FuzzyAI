from typing import Any, Optional

from pydantic import BaseModel

from fuzzyai.db.mongodb import MongoDocument


class AttackResultEntry(BaseModel):
    """
    Entry for the attack result.
    """
    original_prompt: str = str()
    current_prompt: str = str()
    response: str = str()
    classifications: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    def __str__(self) -> str:
        return super().__str__() + f"(prompt={self.current_prompt})"
    
class AttackSummary(MongoDocument):
    attack_mode: str = str()
    model: str = str()
    system_prompt: str = str()
    entries: Optional[list[AttackResultEntry]] = []

    def __str__(self) -> str:
        return super().__str__() + f"(entries={self.entries})"
