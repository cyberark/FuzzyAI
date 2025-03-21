from enum import Enum


class LLMRole(str, Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    USER = "user"

class EnvironmentVariables(Enum):
    OPENAI_API_KEY = "OPENAI_API_KEY"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    GEMINI_API_KEY = "GEMINI_API_KEY"
    AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
    AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
    AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
    DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
    AI21_API_KEY = "AI21_API_KEY"