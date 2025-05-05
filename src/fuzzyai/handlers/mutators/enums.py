from enum import Enum


class MutatorType(str, Enum):
    """
    Enum for breaking the attack.
    """
    REPHRASE = "rephrase"
    RETOKENIZE = "retokenize"
    SUMMARIZE = "summarize"
    FINETUNED_SUMMARIZE = "finetuned_summarize"
    RANDROP = "randrop"
    MIGHT_BE_HARMFUL = "might_be_harmful"