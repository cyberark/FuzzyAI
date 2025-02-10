from enum import Enum


class Classifier(str, Enum):
    COMMITTEE = "com"
    COSINE_SIMILARITY = "cos"
    LLM_RESPONSE_SENTIMENT = "res"
    DISAPPROVAL = "dis"
    GENERIC_LLM = "gen"
    HARMFUL_LLM = "har"
    HARM_SCORE_LLM = "sco"
    OPENAI_MODERATION = "oai"
    AWS_GUARDRAILS = "agr"
    AWS_BEDROCK = "bed"
    AZURE_CONTENT_SAFETY = "acs"
    RATING = "rat"
    SENTIMENT = "sen"
    OBVIOUS = "obv"
