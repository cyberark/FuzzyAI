from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fuzzyai.db.mongodb import MongoDocument
from fuzzyai.handlers.attacks.models import AttackSummary


class PromptEntry(BaseModel):
    original_prompt: str = str()
    original_response: str = str()
    harmful_prompt: str = str()
    harmful_response: str = str()
    classifications: dict[str, Any] = {}
    
    model_config = ConfigDict(extra='allow')

class FuzzerResultModelEntry(BaseModel):
    name: str = str()
    harmful_prompts_count: int = 0
    failed_prompts_count: int = 0
    harmful_prompts: list[PromptEntry] = []
    failed_prompts: list[PromptEntry] = []

    @model_validator(mode='after')
    def calculate_counts(self) -> Any:
        self.harmful_prompts_count = len(self.harmful_prompts)
        self.failed_prompts_count = len(self.failed_prompts)
        return self
    
class FuzzerResultEntry(BaseModel):
    attack_mode: str = str()
    total_prompts_count: int = 0
    models: list[FuzzerResultModelEntry] = []
    success_rate: int = 0

    @model_validator(mode='after')
    def calculate_counters(self) -> Any:
        total_prompts_count = 0
        total_failed_prompts_count = 0

        for m in self.models:
            total_failed_prompts_count += m.failed_prompts_count
            total_prompts_count += m.harmful_prompts_count + m.failed_prompts_count
        
        self.total_prompts_count = total_prompts_count
        self.success_rate = int((total_prompts_count - total_failed_prompts_count) / total_prompts_count * 100) if total_prompts_count > 0 else 0

        return self  


class FuzzerResult(MongoDocument):
    attack_id: str = Field(default_factory=lambda: str(uuid4()))
    attacking_techniques: Optional[list[FuzzerResultEntry]] = []

    @classmethod
    def from_attack_summary(cls, attack_id: str, attack_summaries: list[AttackSummary]) -> 'FuzzerResult':
        attacking_techniques: list[FuzzerResultEntry] = []

        for attack_summary in attack_summaries:
            prompt_entries = [
                PromptEntry(
                    original_prompt=entry.original_prompt,
                    original_response=entry.extra.get('original_response', ''),
                    harmful_prompt=entry.current_prompt,
                    harmful_response=entry.response,
                    classifications=entry.classifications
                ) for entry in attack_summary.entries # type: ignore
            ]

            harmful_prompts = [entry for entry in prompt_entries if 1 in entry.classifications.values()]
            failed_prompts = [entry for entry in prompt_entries if all(x == 0 for x in entry.classifications.values()) and entry.original_prompt not in [x.original_prompt for x in harmful_prompts]]

            # Assuming prompt_entries is the list of PromptEntry objects
            unique_failed_prompts = {}
            for entry in prompt_entries:
                if all(x == 0 for x in entry.classifications.values()):
                    unique_failed_prompts[entry.original_prompt] = entry

            # Convert the dictionary back to a list
            failed_prompts = list(unique_failed_prompts.values())

            models = [
                FuzzerResultModelEntry(
                    name=attack_summary.model,
                    harmful_prompts=harmful_prompts,
                    failed_prompts=failed_prompts
                )
            ]
            # Find an object in attack_techinques with the same attack_mode
            # If it exists, append the models to the existing object
            # If it doesn't exist, create a new object with the attack_mode and models
            for at in attacking_techniques:
                if at.attack_mode == attack_summary.attack_mode:
                    at.models.extend(models)
                    break
            else:
                attacking_techniques.append(
                    FuzzerResultEntry(
                        attack_mode=attack_summary.attack_mode,
                        models=models
                    )
                )
        return cls(attack_id=attack_id, attacking_techniques=attacking_techniques)