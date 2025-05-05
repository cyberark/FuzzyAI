import logging
import string
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union

from fuzzyai.llm.models import BaseLLMProviderResponse

if TYPE_CHECKING:
    from fuzzyai.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class FuzzRunnableProto(Protocol):
    async def run(self, **kwargs: Any) -> str:
        ...

class BaseFuzzRunnable(FuzzRunnableProto):
    async def run(self, **kwargs: Any) -> str:
        raise NotImplementedError
    
    def __or__(self, other: Union['FuzzChain', 'FuzzNode', 'BaseLLMProvider']) -> 'FuzzChain':
        classname = other.__class__.__name__
        if classname == 'FuzzChain':
            return FuzzChain([self, *other._nodes]) # type: ignore
        elif classname == 'FuzzNode':
            return FuzzChain([self, other]) # type: ignore
        elif classname == 'BaseLLMProvider':
            return FuzzChain([self, FuzzNode(other, "{input}")]) # type: ignore
        
        raise ValueError(f"Invalid type for __value: {classname}")

class FuzzNode(BaseFuzzRunnable):
    def __init__(self, llm: 'BaseLLMProvider', prompt: str) -> None:
        self._llm: 'BaseLLMProvider' = llm
        self._prompt = prompt

    async def run(self, **kwargs: Any) -> str:
        full_prompt = self._prompt
        if kwargs is not None and len(kwargs) > 0:
            parsed = string.Formatter().parse(self._prompt)
            for _, field_name, _, _ in parsed:
                if field_name is not None and field_name in kwargs:
                    full_prompt = full_prompt.format(**{field_name: kwargs[field_name]})

        response: Optional[BaseLLMProviderResponse] = await self._llm.generate(full_prompt)
        return response.response if response is not None else ""
    
    def __repr__(self) -> str:
        return f"FuzzNode({self._llm}, {self._prompt})"
    
class FuzzChain(BaseFuzzRunnable):
    def __init__(self, nodes: Optional[list[FuzzNode]] = None) -> None:
        self._nodes: list[FuzzNode] = []

        if nodes is not None:
            self._nodes.extend(nodes)
    
    async def run(self, **kwargs: Any) -> str:
        nodes = reversed(self._nodes)
        for node in nodes:
            logger.debug(f"Running node:\n{node} with kwargs '{kwargs}'")
            response = await node.run(**kwargs)
            logger.debug(f"Node result:\n{response}")
            kwargs = {"input": response}
        return kwargs['input'] # type: ignore