import json
import logging
from typing import Any

from fuzzyai.handlers.attacks.base import attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.taxonomy.handler import TaxonomyParaphraser

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY_MODEL = "openai/gpt-4o"

@attack_handler_fm.flavor(FuzzerAttackMode.HISTORY_FRAMING)
class HistoryFramingAttackHandler(TaxonomyParaphraser):
    """
    History framing attack technique
    """
    def __init__(self, **extra: Any) -> None:
        super().__init__(**extra)

        self._extra_args.taxonomy_model = DEFAULT_TAXONOMY_MODEL

    def _load_taxonomy_dataset(self) -> None:
        with open('resources/persuasion_taxonomy.jsonl', 'r') as f:
            data = f.read()

        self._taxonomies = [json.loads(jline) for jline in data.splitlines()][:1]
    
    @classmethod
    def default_auxiliary_models(cls) -> list[str]:
        return [DEFAULT_TAXONOMY_MODEL]

        