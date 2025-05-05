import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional, overload
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from fuzzyai.consts import DATETIME_FORMAT
from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzyai.handlers.classifiers.base import BaseClassifier
from fuzzyai.handlers.db.adv_attacks import AdversarialAttacksHandler
from fuzzyai.handlers.db.adv_prompts import AdversarialPromptDTO, AdversarialPromptsHandler
from fuzzyai.handlers.db.adv_suffixes import AdversarialSuffixesHandler
from fuzzyai.llm.providers.base import BaseLLMProvider
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.models.fuzzer_result import FuzzerResult
from fuzzyai.utils.utils import llm_provider_factory

logger = logging.getLogger(__name__)


class Fuzzer:
    def __init__(self, db_address: str, cleanup: bool = True, **extra: Any) -> None:
        """
        Initialize the Fuzzer class.

        Args:
            db_address (str): Address of the database.
            **extra (Any): Additional arguments.
        """
        self._extra = extra
        self._classification_batch_size = 150

        self._mongo_client: AsyncIOMotorClient = AsyncIOMotorClient(db_address, 27017)  # ignore: type
        self._adv_prompts_handler = AdversarialPromptsHandler(self._mongo_client)
        self._adv_suffixes_handler = AdversarialSuffixesHandler(self._mongo_client)
        self._adv_attacks_handler = AdversarialAttacksHandler(self._mongo_client)

        self._llms: list[BaseLLMProvider] = []
        self._classifiers: list[BaseClassifier] = []
        
        self._attack_id = str(uuid4())
        self._attack_time = datetime.now().strftime(DATETIME_FORMAT)
        self._cleanup = cleanup
        
        logger.info(f"Initiating Attack ID: {self._attack_id}, Attack Time: {self._attack_time}, DB Address: {db_address}")

    def add_classifier(self, classifier: BaseClassifier, **extra: Any) -> None:
        """
        Add a new classifier.

        Args:
            classifier (BaseClassifier): The classifier to add.
        """
        self._classifiers.append(classifier)

    def add_llm(self, provider_and_model: str, **extra: Any) -> None:
        """
        Add a new LLM provider.

        Args:
            provider_and_model (str): The providel and model to use in the form of provider/model_name.
            model (str): Which model to use with the provided provider.
        """
        if '/' not in provider_and_model:
            raise RuntimeError(f"Model {provider_and_model} not in correct format, please use provider/model_name format")

        provider_name, model = provider_and_model.split('/', 1)

        if provider_name not in LLMProvider.__members__.values():
            raise RuntimeError(f"Provider {provider_name} not found")

        is_valid_model: bool = True

        if provider_name == LLMProvider.LOCAL:
            is_valid_model = os.path.isdir(model)
        elif provider_name == LLMProvider.REST:
            is_valid_model = os.path.isfile(model)

        if not is_valid_model:
            logger.error(f"Model {model} not found for {provider_name.lower()} provider")
            return

        logger.debug(f"Adding {model} model on {provider_name}")
        llm = llm_provider_factory(provider=LLMProvider(provider_name), model=model, **extra)
        self._llms.append(llm)

    def get_llm(self, model: str) -> Optional[BaseLLMProvider]:
        """
        Get the LLM provider for the given model.

        Args:
            model (str): The model to get the provider for.

        Returns:
            Optional[BaseLLMProvider]: The LLM provider.
        """
        for llm in self._llms:
            if llm.qualified_model_name == model:
                return llm
        return None
    
    @overload
    async def fuzz(self, attack_modes: list[FuzzerAttackMode], model: list[str], prompts: str, **extra: Any) -> tuple[FuzzerResult, list[AttackSummary]]:
        """
        Fuzz a single prompt.

        Args:
            prompt (str): The prompt to fuzz.
            model (list[str]): The models to attack. Each entry here must first be added using the add_llm method.
            attack_modes (list[FuzzerAttackMode]): The attack modes.

        Returns:
            tuple[FuzzerResult, list[AttackSummary]]: Summarized results + raw results.
        """
        ...

    @overload
    async def fuzz(self, attack_modes: list[FuzzerAttackMode], model: list[str], **extra: Any) -> tuple[FuzzerResult, list[AttackSummary]]:
        """
        Fuzz prompts from the database

        Args:
            attack_modes (list[FuzzerAttackMode]): The attack modes.
            model (list[str]): The models to attack. Each entry here must first be added using the add_llm method.

        Returns:
            tuple[FuzzerResult, list[AttackSummary]]: Summarized results + raw results.
        """
        ...
    
    async def fuzz(self, attack_modes: list[FuzzerAttackMode], model: list[str], prompts: Optional[list[str] | str] = None, **extra: Any) -> tuple[FuzzerResult, list[AttackSummary]]:
        """
        Fuzz multiple prompts.

        Args:
            attack_modes (list[FuzzerAttackMode]): The attack modes.
            model (list[str]): The models to attack. Each entry here must first be added using the add_llm method.
            prompts (list[AdversarialPromptDTO]): The prompts to fuzz.

        Returns:
            tuple[FuzzerResult, list[AttackSummary]]: Summarized results + raw results.
        """
        prompts_dto: list[AdversarialPromptDTO]  = []

        if prompts is None:
            prompts_dto = await self._adv_prompts_handler.retrieve()
        elif isinstance(prompts, str):
            prompts_dto = [AdversarialPromptDTO(prompt=prompts)]
        elif isinstance(prompts, list):
            prompts_dto = [AdversarialPromptDTO(prompt=prompt) for prompt in prompts]

        return await self._fuzz(prompts_dto, model, attack_modes, **extra)
    
    async def cleanup(self) -> None:
        """
        Cleanup the fuzzer.
        """
        await asyncio.gather(*[llm.close() for llm in self._llms])

    def _attack_technique_factory(self, attack_mode: FuzzerAttackMode, model: str, **extra: Any) -> BaseAttackTechniqueHandlerProto:
        """
        Factory method to create an instance of the attack technique handler.

        Args:
            attack_mode (FuzzerAttackMode): The attack mode.
            model (str): The model to attack.
            **extra (Any): Additional arguments for the attack technique handler.

        Returns:
            BaseAttackTechniqueHandler: An instance of the attack technique handler.
        """
        handler_cls: type[BaseAttackTechniqueHandler[BaseModel]] = attack_handler_fm[attack_mode]
        if (auxiliary_models := handler_cls.default_auxiliary_models()) is not None:
            for auxiliary_model in auxiliary_models:
                if not any(llm.qualified_model_name == auxiliary_model for llm in self._llms):
                    logger.info(f"Attack mode {attack_mode} defines a default auxiliary model {auxiliary_model}. Automatically adding it - see wiki for more details.")
                    self.add_llm(auxiliary_model)

        return attack_handler_fm[attack_mode](llms=self._llms, model=model,
                                              classifiers=self._classifiers, **extra)
    
    async def _fuzz(self, prompts: list[AdversarialPromptDTO], models: list[str],
                    attack_modes: list[FuzzerAttackMode], **extra: Any) -> tuple[FuzzerResult, list[AttackSummary]]:
        """
        Fuzz the given prompts.

        Args:
            prompts (list[AdversarialPromptDTO]): The prompts to fuzz.
            models (list[str]): The models to attack.
            attack_mode (FuzzerAttackMode): The attack mode.
        """
        raw_results: list[AttackSummary] = []
        attack_handler: Optional[BaseAttackTechniqueHandlerProto] = None
        
        logger.info('Starting fuzzer...')
        
        start_time = time.time()

        # Verify that all models has been added by add_llm
        for model in models:
            if not any(llm.qualified_model_name == model for llm in self._llms):
                logger.error(f"Model {model} has not been added to the fuzzer, skipping...")
                continue

            for attack_mode in attack_modes:
                logger.info(f'Attacking {len(prompts)} prompts with attack mode: {attack_mode} for model: {model}...')
                extra.update(**self._extra)
                attack_handler = self._attack_technique_factory(attack_mode, model=model, **extra)

                attack_result: Optional[AttackSummary] = await attack_handler.attack(prompts)
                if attack_result:
                    attack_result.attack_mode = attack_mode
                    attack_result.model = model
                    attack_result.system_prompt = extra.get('system_prompt', 'No system prompt set')
                    logger.info(f'Finished attacking {len(prompts)} prompts for attack mode {attack_mode}')
                    raw_results.append(attack_result)
                else:
                    logger.error(f'Failed to attack {len(prompts)} prompts for attack mode {attack_mode}')
        
        logger.info('Done, took %s seconds', time.time() - start_time)
        
        if attack_handler is not None and self._cleanup:
            await asyncio.gather(attack_handler.close())

        report = FuzzerResult.from_attack_summary(self._attack_id, raw_results)
        return report, raw_results
    



