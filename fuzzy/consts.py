from typing import Final

DATABASE_NAME: Final[str] = "fuzzy"
PROMPT_COLLECTION_NAME: Final[str] = "adversarial_prompts"
ADVERSARIAL_SUFFIX_COLLECTION_NAME: Final[str] = "adversarial_suffixes"
ADVERSARIAL_ATTACK_COLLECTION_NAME: Final[str] = "adversarial_attacks"
CLASSIFICATIONS_COLLECTION_NAME: Final[str] = "classifications"
TAXONOMIES_COLLECTION_NAME: Final[str] = "taxonomies"
GCG_ATTACK_COLLECTION_NAME: Final[str] = "gcg_attack"
GENETIC_ATTACK_COLLECTION_NAME: Final[str] = "genetic_attack"
BENCHMARKS_COLLECTION_NAME: Final[str] = "benchmarks"
FUZZER_REPORT_COLLECTION_NAME: Final[str] = "reports"
WIKI_LINK: Final[str] = "https://github.com/cyberark/FuzzyAI/wiki"

# Parameters
PARAMETER_MAX_TOKENS: Final[str] = "max_tokens"

# Misc
DATETIME_FORMAT: Final[str] = "%Y/%m/%d::%H:%M:%S"

# Mongo
ID_FIELD: Final[str] = "_id"
MONGO_OPERATOR_GT: Final[str] = "$gt"

# Modeling
FIELD_ID: Final[str] = "_id"
FIELD_NAME_SUFFIX: Final[str] = "suffix"
FIELD_NAME_PROMPT: Final[str] = "prompt"
FIELD_NAME_ATTACK_ID: Final[str] = "attack_id"
FIELD_NAME_ATTACK_TIME: Final[str] = "attack_time"
FIELD_NAME_ATTACK_REQUEST: Final[str] = "request"
FIELD_NAME_ATTACK_RESPONSE: Final[str] = "response"
FIELD_NAME_ATTACK_LLM_TYPE: Final[str] = "llm_type"
FIELD_NAME_ATTACK_LABEL: Final[str] = "label"
FIELD_NAME_ATTACK_SENTIMENT_ANALYSIS: Final[str] = "sentiment_analysis"

# Ollama
OLLAMA_BASE_PORT: Final[int] = 11434
DEFAULT_OPEN_SOURCE_MODEL: Final[str] = "ollama/qwen2.5"

# LLM Providers extra
LLM_PROVIDER_EXTRA_SEED: Final[str] = "seed"
LLM_PROVIDER_EXTRA_SYSTEM_PROMPT: Final[str] = "system_prompt"
