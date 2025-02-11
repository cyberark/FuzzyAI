import json
import logging
import os
from typing import Any, Callable, Coroutine, Optional, Union

import aiohttp
import requests
from jsonpath_ng import parse

from fuzzy.llm.models import BaseLLMProviderResponse
from fuzzy.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from fuzzy.llm.providers.enums import LLMProvider
from fuzzy.llm.providers.rest.utils import parse_http_request

PROMPT_TOKEN = "<PROMPT>"

logger = logging.getLogger(__name__)

class RestProviderException(BaseLLMProviderException):
    pass

@llm_provider_fm.flavor(LLMProvider.REST)
class RestProvider(BaseLLMProvider):
    """
    A provider that uses a REST API to communicate with a language model.
    This provider is used to communicate with any language model that exposes a REST API, or a REST API which utilizes an LLM behind the scene.

    caveats:
        * REST API request must be in JSON format.
        * REST API response must be in JSON format.
        * We assume that the raw http request file is in the following format:
            ```
            METHOD /path HTTP/1.1
            Content-type: application/json
            key: value
            key: value
            
            {
                "key": "value"
            }
            ``` 

    Args:
        model (str): The path to the raw HTTP request file.
        host (str): The host of the REST API.
        response_jsonpath (str): The JSONPath to extract the response from the HTTP response. (default: "$.response").
        prompt_token (str): The token to be replaced with the prompt in the HTTP request body. (default: "<PROMPT>")
        scheme (str): The scheme of the REST API (default: "https").
        port (int): The port of the REST API (default: 443).
        **extra (Any): Additional arguments to be passed to the BaseLLMProvider constructor.
    """
    def __init__(self, model: str, host: Optional[str] = None, response_jsonpath: str = "$.response", 
                 prompt_token: str = PROMPT_TOKEN, scheme: str = "https", port: int = 443, **extra: Any):
        super().__init__(model=model, **extra)

        if any(x is None for x in [host, response_jsonpath]):
            raise RuntimeError("host, and response_jsonpath must be provided using -e flag.")
        
        self._method: str = str()
        self._path: str = str()
        self._headers: dict[str, Any] = dict()
        self._body: str = str()

        self._prompt_token = prompt_token
        self._response_jsonpath = response_jsonpath
        self._parse_http_file(model)

        self._url = f"{scheme}://{host}:{port}{self._path}"

        # Initialize both async and sync sessions
        self._session = aiohttp.ClientSession(headers=self._headers)
        self._sync_session = requests.Session()
        self._sync_session.headers.update(self._headers)

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return "<Path to raw HTTP request file>"
    
    def _prepare_request_payload(self, prompt: str) -> str:
        """
        Prepare the request payload by replacing the prompt token.
        
        Args:
            prompt (str): The input prompt to generate a response.
            
        Returns:
            str: The prepared request payload
        """
        sanitized_prompt = json.dumps(prompt)[:-1][1:]
        return self._body.replace(self._prompt_token, sanitized_prompt)
    
    def _parse_http_file(self, raw_http_file: str) -> None:
        """
        Parse the raw HTTP file to extract the method, url, headers, and body.

        Args:
            raw_http_file (str): The path to the raw HTTP file.

        """
        # Check that the file exists
        if not os.path.exists(raw_http_file):
            raise RestProviderException(f"HTTP file not found: {raw_http_file}")
        
        parsed_http = parse_http_request(raw_http_file)
        logger.debug("Parsed HTTP: %s", parsed_http)

        self._method = parsed_http["method"]
        self._path = parsed_http["path"]
        if not self._path.startswith("/"):
            self._path = f"/{self._path}"

        self._headers = parsed_http["headers"]
        # Ditch Content-length header
        for header in ["Content-Length", "content-length"]:
            self._headers.pop(header, None)

        self._body = parsed_http["body"]
    
    def _process_response(self, raw_response: dict[str, Any]) -> Optional[BaseLLMProviderResponse]:
        """
        Process the response from the API using JSONPath.
        
        Args:
            raw_response (dict): The raw response from the API
            
        Returns:
            Optional[BaseLLMProviderResponse]: The processed response
        """
        jsonpath_expr = parse(self._response_jsonpath)
        logger.debug("Raw response: %s", raw_response)
        logger.debug("Extracting response using JSONPath: %s", self._response_jsonpath)

        result = [match.value for match in jsonpath_expr.find(raw_response)]
        if result:
            return BaseLLMProviderResponse(response=result[0])
        logger.warning("No response found in the JSONPath: %s", self._response_jsonpath)
        return None
    
    async def generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        """
        Generates a response from the language model using async REST API.

        Args:
            prompt (str): The input prompt to generate a response.
            **extra (Any): Additional arguments to be passed to the REST API.

        Returns:
            Optional[BaseLLMProviderResponse]: The generated response.
        """
        logger.debug("Generating prompt (async): %s", prompt)
        try:
            method: Callable[..., Coroutine[Any, Any, Any]] = getattr(self._session, self._method.lower())
            payload = self._prepare_request_payload(prompt)          
            http_response = await method(url=self._url, json=json.loads(payload))
            http_response.raise_for_status()
            
            raw_response = await http_response.json()
            return self._process_response(raw_response)
            
        except Exception as e:
            logger.error("Error generating response: %s", e)
            raise RestProviderException(f"Error generating prompt: {e}")

    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        """
        Generates a response from the language model using synchronous REST API.

        Args:
            prompt (str): The input prompt to generate a response.
            **extra (Any): Additional arguments to be passed to the REST API.

        Returns:
            Optional[BaseLLMProviderResponse]: The generated response.
        """
        logger.debug("Generating prompt (sync): %s", prompt)
        try:
            method: Callable[..., requests.Response] = getattr(self._sync_session, self._method.lower())
            payload = self._prepare_request_payload(prompt)
            
            http_response = method(url=self._url, json=json.loads(payload))
            http_response.raise_for_status()
            
            raw_response = http_response.json()
            return self._process_response(raw_response)
            
        except Exception as e:
            logger.error("Error generating response: %s", e)
            raise RestProviderException(f"Error generating prompt: {e}")

    async def close(self) -> None:
        await self._session.close()
        self._sync_session.close()
    
    async def chat(self, messages: list[BaseLLMMessage], **extra: Any) -> BaseLLMProviderResponse | None:
        raise Exception("Chat is not supported for REST providers.")
    
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> BaseLLMProviderResponse | None:
        raise Exception("Chat is not supported for REST providers.")
    