import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Generic, Optional, Type, TypeVar, cast

import aiofiles
import aiofiles.os
from pydantic import BaseModel
from pydantic_core import PydanticUndefinedType, ValidationError
from tqdm import tqdm

from fuzzy.handlers.attacks.proto import AttackResultEntry, AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzy.handlers.classifiers.base import BaseClassifier
from fuzzy.handlers.classifiers.enums import Classifier
from fuzzy.handlers.db.adv_prompts import AdversarialPromptDTO
from fuzzy.handlers.response_refinement_handler import RefinementException, ResponseRefinementHandler
from fuzzy.llm.models import BaseLLMProviderResponse
from fuzzy.llm.providers.base import BaseLLMProvider
from fuzzy.utils.flavor_manager import FlavorManager

logger = logging.getLogger(__name__)

T = TypeVar("T")

class NoArgs(BaseModel):
    ...

class AttackBreakWhen(str, Enum):
    """
    Enum for breaking the attack.
    """
    FIRST_COMPLETED = "first_completed"
    ALL_COMPLETED = "all_completed"
    FIRST_RESULT = "first_result"

class BaseAttackTechniqueHandlerException(Exception):
    """
    Base exception for attack techniques.
    """
    ...

class BaseAttackTechniqueHandler(BaseAttackTechniqueHandlerProto, Generic[T]):
    """
    Base class for attack techniques.

    Args:
        llms (list[BaseLLMProvider]): List of LLM providers to use for the attack.
        model (str): Model to use for this specific attack.
        classifiers (list[BaseClassifier], optional): List of classifiers to use for the attack. Defaults to None.
        max_workers (int, optional): Maximum number of workers. Defaults to 1.
        break_when (AttackBreakWhen, optional): When to break the attack. Defaults to AttackBreakWhen.ALL_COMPLETED.
        classifier_model (Optional[str], optional): Defines which model to use when classifying the results. Defaults to None (i.e use same model as the attack).
        attack_id (Optional[str], optional): Attack ID. Defaults to None.
        **extra (Any): Additional arguments.
    """
    def __init__(self, llms: list[BaseLLMProvider],
                 model: str,
                 classifiers: list[BaseClassifier] | None = None, 
                 max_workers: int = 1, 
                 break_when: AttackBreakWhen = AttackBreakWhen.ALL_COMPLETED,
                 classifier_model: Optional[str] = None,
                 attack_id: Optional[str] = None,
                 improve_attempts: int = 0,
                 **extra: Any) -> None:
        self._llms = llms
        self._model = model
        self._classifiers: list[BaseClassifier] = []
        self._classifier_model = classifier_model
        
        for classifier in classifiers or []:
            self._classifiers.append(classifier)
                
        self._max_workers = max_workers
        self._break_when = break_when
        
        self._attack_id = attack_id or uuid.uuid4().hex
        self._completed_prompts: set[str] = set()
        self._progress_bar: Optional[tqdm] = None
        self._params_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._model_queue_map: dict[str, asyncio.Queue[BaseLLMProvider]] = self._create_model_queue_map([x._qualified_model_name for x in llms])
        self._refinement_handler = ResponseRefinementHandler(improve_attempts) if improve_attempts > 0 else None


        for llm in llms:
            self._model_queue_map[llm.qualified_model_name].put_nowait(llm)

        try:
            self._extra_args: T = cast(T, self.extra_args_cls()(**extra))
        except ValidationError as ex:
            logger.error(f"You must provide the following extra arguments: {ex.errors()}", exc_info=True)
            raise RuntimeError("Invalid or missing extra arguments, please use --list-extra or see wiki") from ex
        
        self._extra = extra # Save raw extra, maybe we need to remove this

    """
    Generate the description of the attack handler.

    Returns:
        str: Description of the attack handler (docstring)
    """
    @classmethod
    def description(cls) -> str:
        return cls.__doc__ or "No description available"
    
    """
    Get the extra arguments class for the attack handler. Used for typing.
    Returns:
        Type[BaseModel]: Extra arguments class. (Default: NoArgs)
    """
    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return NoArgs

    """
    Get the extra arguments dictionary for the attack handler.
    Returns:
        dict[str, str] | None: Extra arguments for the attack handler in the form of (arg, help)
    """
    @classmethod
    def extra_args(cls) -> dict[str, str] | None:
        result: dict[str, str] = {}

        for k, v in cls.extra_args_cls().model_fields.items():
            if v.description is not None:
                description = v.description
                if v.default and not isinstance(v.default, PydanticUndefinedType):
                    description += f" (default: {v.default})"
                result[k] = description

        return result

    """
    Close the handler.
    """
    async def close(self) -> None:
        await asyncio.gather(*[provider.close() for provider in self._llms])
    
    """
    Perform the attack using the given prompt.
    Args:
        prompts (list[AdversarialPromptDTO]): Prompts to attack.
    Returns:
        AttackSummary: Adversarial prompts generated by the attack, if any
    """
    async def attack(self, prompts: list[AdversarialPromptDTO]) -> Optional[AttackSummary]:
        pending_tasks = set()
        result = AttackSummary()

        self._verify_supported_models()
        self._verify_supported_classifiers()
        
        attack_params: list[dict[str, Any]] = self._generate_attack_params(prompts)
        logger.info(f"Generated {len(attack_params)} attack params for {len(prompts)} prompts")
        self._progress_bar = tqdm(total=len(attack_params), desc="Attacking")

        # Check for previous attack results
        attack_params = await self._check_previous_attack(result, attack_params)

        for param in attack_params:
            self._params_queue.put_nowait(param)

        await aiofiles.os.makedirs('.out', exist_ok=True)
        results_file_path = Path(".out") / self._attack_id if self._attack_id else Path(".out") / self._attack_id
        results_file = await aiofiles.open(results_file_path, "a", encoding="utf-8")
        
        for i in range(self._max_workers):
            pending_tasks.add(asyncio.create_task(self._consume_attack_params(results_file), name=f"{i}"))

        try:
            while True:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=60)
                for task in done:
                    entry: Optional[list[AttackResultEntry]] = task.result()
                    if entry:
                        result.entries.extend(entry) # type: ignore
                    elif self._break_when == AttackBreakWhen.FIRST_RESULT:
                        break
                
                if result.entries and self._break_when in (AttackBreakWhen.FIRST_COMPLETED, AttackBreakWhen.FIRST_RESULT):
                    for task in pending_tasks:
                        task.cancel()

                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                    break
                
                if not pending_tasks:
                    break
        except Exception as ex:
            logger.error(f"Error during attack: {ex}", exc_info=True)
            raise
        finally:
            self._progress_bar.close()
            await results_file.close()
        
        return result

    """
    Reduce the parameters generated by _generate_attack_params. This is called when a previous attack_id is supplied to the fuzzer.
    Args:
        entries (list[AttackResultEntry]): Previous attack results for the given attack_id. Loaded from a local file.
        attack_params (list[dict[str, Any]]): Attack parameters to reduce, based on the previous results.
    """
    async def _reduce_attack_params(self, entries: list[AttackResultEntry], 
                                    attack_params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [param for param in attack_params if not any(entry.current_prompt == param['prompt'] for entry in entries)]
    
    """
    Check for previous attack results. If a previous attack_id is supplied, this method will load the results from a local file.
    Args:
        result (AttackSummary): Attack summary to store the results in.
        attack_params (list[dict[str, Any]]): Attack parameters to check against.
    Returns:
        list[dict[str, Any]]: Reduced attack parameters to use for the attack.
    """
    async def _check_previous_attack(self, result: AttackSummary, attack_params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._attack_id and await aiofiles.os.path.exists(".out" / Path(self._attack_id)):
            logger.info(f"Found previous execution for attack {self._attack_id}, reducing attack params")
            async with aiofiles.open(".out" / Path(self._attack_id), "r", encoding="utf-8") as f:
                data = await f.read()

            result.entries = [AttackResultEntry.model_validate_json(jline) for jline in data.splitlines()]
            original_count = len(attack_params)
            attack_params = await self._reduce_attack_params(result.entries, attack_params)
            if self._progress_bar:
                self._progress_bar.update(original_count - len(attack_params))
        return attack_params

    """
    Classify the response from the LLM provider.
    Args:
        llm_response (Optional[BaseLLMProviderResponse]): Response from the LLM provider.
    Returns:
        dict[str, Any]: Classification results.
    """
    async def _classify_llm_response(self, llm_response: Optional[BaseLLMProviderResponse], **extra: Any) -> dict[str, Any]:
        if llm_response is None:
            return {}
        
        classifications: dict[str, Any] = {}
        
        if self._classifiers:
            target_model = self._classifier_model or self._model
            async with self._borrow(target_model) as llm:
                if llm_response.response:
                    tasks = [
                        asyncio.create_task(self._classify(classifier, llm_response, llm, **extra))
                        for classifier in self._classifiers
                    ]
                    results = await asyncio.gather(*tasks)
                    for result in results:
                        classifications.update(result)
        
        return classifications
    
    async def _classify(self, classifier: BaseClassifier, llm_response: BaseLLMProviderResponse, llm: BaseLLMProvider, **extra: Any) -> dict[str, int]:
        classifier_result = await classifier.classify(text=llm_response.response, llm=llm, **extra)
        return {classifier.name: 1 if classifier.is_jailbreak(classifier_result) else 0}
    
    def _get_classifier(self, classifier_type: Classifier) -> Optional[BaseClassifier]:
        return next((classifier for classifier in self._classifiers if classifier._classifier_type == classifier_type), None)
    
    """
    Borrow a provider from the queue for a specific model.
    Args:
        qualified_model_name (str): Model to borrow, in the provider/model format.
    Returns:
        BaseLLMProvider: Borrowed LLM provider. 
    """
    @asynccontextmanager
    async def _borrow(self, qualified_model_name: str) -> AsyncIterator[BaseLLMProvider]:
        if qualified_model_name not in self._model_queue_map:
            raise BaseAttackTechniqueHandlerException(f"Model {qualified_model_name} not added to the fuzzer!")
        
        queue: asyncio.Queue[BaseLLMProvider] = self._model_queue_map[qualified_model_name]
        provider: BaseLLMProvider = await queue.get()
        logger.debug(f"Task {asyncio.current_task().get_name()} borrowed provider: {provider}") # type: ignore
        yield provider
        queue.task_done()
        queue.put_nowait(provider)

    """
    Borrow any provider from the queue. 
    Returns:
        BaseLLMProvider: Borrowed LLM provider.
    """
    @asynccontextmanager
    async def _borrow_any(self) -> AsyncIterator[BaseLLMProvider]:
        provider: BaseLLMProvider = self._llms.pop()
        logger.debug(f"Task {asyncio.current_task().get_name()} borrowed provider: {provider}") # type: ignore
        yield provider
        self._llms.append(provider)

    """
    Worker logic: Consume the attack parameters from the queue.
    Args:
        results_file (Any): File to write the results to.
    Returns:
        Optional[list[AttackResultEntry]]: Attack results.
    """
    async def _consume_attack_params(self, results_file: Any) -> Optional[list[AttackResultEntry]]:
        results: list[AttackResultEntry] = []
        initial = True
        param: dict[str, Any] = {}

        try:
            while not self._params_queue.empty():
                if not initial and self._progress_bar:
                    self._progress_bar.update()

                initial = False
                
                param = self._params_queue.get_nowait()
                logger.debug(f"Task {asyncio.current_task().get_name()} consuming param: {param['prompt']}") # type: ignore

                if param['prompt'] in self._completed_prompts:
                    logger.debug("Skipping completed prompt")
                    self._params_queue.task_done()
                    continue

                entry = await self._attack(**param)
                if entry:
                    # Make sure we haven't stored a result for this prompt already
                    if entry.original_prompt in self._completed_prompts:
                        logger.debug("Skipping completed prompt")
                        self._params_queue.task_done()
                        continue

                    if entry.classifications and 1 in entry.classifications.values():
                        logger.debug("Jailbreak detected!")
                        self._completed_prompts.add(entry.original_prompt)
                        try:
                            if self._refinement_handler is not None:
                                logger.info("Refining jailbroken response...")
                                async with self._borrow(self._model) as llm:
                                    improved_answers: list[str] = await self._refinement_handler.refine_response(model=llm, original_prompt=entry.current_prompt, response=entry.response)
                                    entry.extra['improved_answers'] = improved_answers
                                    logger.info("Response refined successfully")
                                    logger.debug(f"Refined result: {improved_answers}")
                        except RefinementException as e:
                            logger.error(f"Error refining response: {e}")
                            raise RefinementException(f"Failed to refine answer: {e}")
                    results.append(entry)
                    await results_file.writelines(entry.model_dump_json())
                    await results_file.flush()

                self._params_queue.task_done()  

            if self._progress_bar:
                self._progress_bar.update() 
        except asyncio.CancelledError:
            logger.debug(f"Task {asyncio.current_task().get_name()} was cancelled, returning {len(results)} results") # type: ignore
        except Exception as ex:
            logger.debug(f"Error during attack: {ex} for task {asyncio.current_task().get_name()}", exc_info=True) # type: ignore
            logger.error(f"Error during attack: {ex} for task {asyncio.current_task().get_name()}", exc_info=False) # type: ignore
            # Requeue the item
            if param:
                self._params_queue.put_nowait(param)
        return results
    
    def _generate_attack_params(self, prompts: list[AdversarialPromptDTO]) -> list[dict[str, Any]]:
        return [{"prompt": prompt.prompt} for prompt in prompts]
    
    def _verify_supported_models(self) -> None:
        ...

    def _verify_supported_classifiers(self) -> None:
        ...
    
    def _create_model_queue_map(self, model_types: list[str]) -> dict[str, asyncio.Queue[BaseLLMProvider]]:
        used_models: set[str] = set(model_types)
        return {k: asyncio.Queue() for k in used_models}
    
attack_handler_fm: FlavorManager[str, Type[BaseAttackTechniqueHandler[BaseModel]]] = FlavorManager()