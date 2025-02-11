import abc
import inspect
import logging
from functools import wraps
from typing import Any, Type, TypeVar

from fuzzy.handlers.classifiers.enums import Classifier
from fuzzy.handlers.classifiers.utils import remove_cot, replace_nth
from fuzzy.utils.flavor_manager import FlavorManager, TypedFlavorManager

KeyT = TypeVar('KeyT')
ValT = TypeVar('ValT')

logger = logging.getLogger(__name__)

def pre_classify_hook(func: Any) -> Any:
    @wraps(func)
    async def async_wrapper(self, *args: Any, **kwargs: Any) -> Any: # type: ignore
        args, kwargs = self._preprocess(*args, **kwargs)
        return await func(self, *args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        args, kwargs = self._preprocess(*args, **kwargs)
        return func(self, *args, **kwargs)
    
    # Return appropriate wrapper based on whether the function is async or not
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

class BaseClassifier(abc.ABC):
    """
    Base class for classifiers.
    """
    def __init__(self, classifier: Classifier, truncate_cot: bool, **extra: Any) -> None:
        self._name = str()
        self._classifier_type = classifier
        self._truncate_cot = truncate_cot

    """
    Run the classification logic. Explicitly extract the needed parameter in the function signature.
    """
    @pre_classify_hook
    async def classify(self, *args: Any, **extra: Any) -> Any:
        return await self._classify(*args, **extra)

    @pre_classify_hook
    def sync_classify(self, *args: Any, **extra: Any) -> Any:
        return self._sync_classify(*args, **extra)

    @abc.abstractmethod
    async def _classify(self, *args: Any, **extra: Any) -> Any: ...

    @abc.abstractmethod
    def _sync_classify(self, *args: Any, **extra: Any) -> Any: ...
    
    """
    In most cases, the classifier will return a boolean value. This method will be used to determine if the value is a jailbreak or not.
    Override to provide custom logic.
    """
    def is_jailbreak(self, value: Any) -> bool:
        return bool(value)

    def _preprocess(self, *args: Any, **extra: Any) -> tuple[tuple[Any, ...], dict[Any, Any]]:
        if not self._truncate_cot:
            return args, extra
        
        logger.debug("Preprocessing classifier arguments")
        n_args = args

        if 'text' not in extra:
            method_signature = inspect.signature(self._classify)
            if "text" not in method_signature.parameters:
                logger.debug("text parameter is not defined on classify method signature, won't preprocess")
                return n_args, extra
            
            text_param_index = list(method_signature.parameters.keys()).index('text')
            n_args = replace_nth(n_args, text_param_index, remove_cot(n_args[text_param_index]))
        else:
            extra['text'] = remove_cot(extra['text'])

        return n_args, extra
    
    @property
    def name(self) -> str:
        return self._name 

    @classmethod
    def description(cls) -> str:
        return cls.__doc__ or "No description available"
    
    @classmethod
    def requires_llm(cls) -> bool:
        method_signature = inspect.signature(cls.classify)
        if "llm" not in method_signature.parameters:
            return False
        
        param = method_signature.parameters['llm']
        return bool(param.default == inspect.Parameter.empty)
    
class ClassifierFlavorManager(TypedFlavorManager[KeyT, ValT]):
    def __init__(self) -> None:
        super().__init__()

    def kwargs_type_parameter_name(self) -> str:
        return "classifier"
    
classifiers_fm: ClassifierFlavorManager[str, Type[BaseClassifier]] = ClassifierFlavorManager()
