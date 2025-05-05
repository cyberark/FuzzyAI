from enum import Enum, auto


class PersuasiveActor(str, Enum):
    ATTACKER = auto()
    TARGET = auto()
    JUDGE = auto()
