    
import re
from typing import Any


def replace_nth(t: tuple[Any, ...], n: int, new_value: Any) -> tuple[Any, ...]:
        """Replace the n-th element (0-based) in a tuple with new_value."""
        if not (0 <= n < len(t)):
            raise IndexError("Tuple index out of range")

        return t[:n] + (new_value,) + t[n+1:]

def remove_cot(data: str) -> str:
    return re.sub(r'<think>.*?</think>\s*', '', data, flags=re.DOTALL)