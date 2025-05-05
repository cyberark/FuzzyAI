from enum import Enum


class BedrockModelFamily(Enum):
    """The model family of a Bedrock model."""
    ANTHROPIC = "anthropic"
    META = "meta"
    TITAN = "titan"
    