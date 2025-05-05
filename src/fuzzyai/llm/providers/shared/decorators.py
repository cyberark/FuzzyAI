import functools
from typing import Any, Callable, TypeVar

T = TypeVar('T', bound='RestApiProvider')

class RestApiProvider:
    _base_url: str
    
def api_endpoint(endpoint: str) -> Callable: # type: ignore
    def decorator(func: Callable) -> Callable: # type: ignore
        @functools.wraps(func)
        async def wrapper(self: T, *args, **kwargs) -> Any: # type: ignore
            url = f"{self._base_url}{endpoint}"
            if "url" in kwargs:
                url = kwargs.pop("url")  # Use provided URL if available
            else:
                kwargs["url"] = url  # Add default URL to kwargs if not provided

            return await func(self, *args, **kwargs)

        return wrapper

    return decorator

def sync_api_endpoint(endpoint: str) -> Callable: # type: ignore
    def decorator(func: Callable) -> Callable: # type: ignore
        @functools.wraps(func)
        def wrapper(self: T, *args, **kwargs) -> Any: # type: ignore
            url = f"{self._base_url}{endpoint}"
            if "url" in kwargs:
                url = kwargs.pop("url")  # Use provided URL if available
            else:
                kwargs["url"] = url  # Add default URL to kwargs if not provided

            return func(self, *args, **kwargs)

        return wrapper

    return decorator