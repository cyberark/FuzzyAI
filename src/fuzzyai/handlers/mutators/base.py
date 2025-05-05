import abc
from typing import Any, Type

from fuzzyai.utils.flavor_manager import FlavorManager


class BaseMutator(abc.ABC):
    """
    Base class for mutators. Mutators recieve input and apply a transformation to it.
    """
    def __init__(self, name: str, **extra: Any) -> None:
        self._name = name
        self._extra = extra

    """
    Apply the mutation logic.
    Args:
        prompt (str): The prompt to mutate.
    Returns:
        str: The mutated prompt.
    """
    @abc.abstractmethod
    async def mutate(self, prompt: str) -> str:
        ...

    def get_name(self) -> str:
        return self._name
    
    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        return await self.mutate(*args, **kwds)
    
mutators_fm: FlavorManager[str, Type[BaseMutator]] = FlavorManager()