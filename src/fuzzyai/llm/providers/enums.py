from enum import Enum


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    AZURE = "azure"
    OPENAI = "openai"
    AWS_BEDROCK = "aws-bedrock"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AI21 = "ai21"
    DEEPSEEK = "deepseek"
    LOCAL = "local"
    LOCAL_LLAMA2 = "local-llama2"
    REST = "rest"


class LLMProviderExtraParams(str, Enum):
    MAX_LENGTH = "max_length"
    APPEND_LAST_RESPONSE = "append_last_response"
