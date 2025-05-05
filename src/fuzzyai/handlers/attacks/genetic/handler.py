# type: ignore
import logging
from typing import Any, Optional, Union

import numpy as np
import pygad
from pydantic import BaseModel

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.classifiers.cosine_similarity.handler import CosineSimilarityClassifier
from fuzzyai.handlers.classifiers.disapproval.handler import DisapprovalClassifier
from fuzzyai.handlers.db.adv_prompts import AdversarialPromptDTO
from fuzzyai.handlers.tokenizers.handler import TokensHandler
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.enums import LLMProviderExtraParams
from fuzzyai.llm.providers.local.llama2_llm import Llama2Provider
from fuzzyai.llm.providers.openai.openai import OpenAIProvider

logger = logging.getLogger(__name__)

MAX_RETRY_COUNT = 3
SPLIT_TOKEN = ";"


@attack_handler_fm.flavor(FuzzerAttackMode.GENETIC)
class GeneticAttackTechniqueHandler(BaseAttackTechniqueHandler[BaseModel]):
    """
    Genetic attack (https://arxiv.org/abs/2309.01446)
    """

    def __init__(self, **extra: Any):
        super().__init__(**extra)

        self._population_seed = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        self._suffixes_batch_size = 16
        self._fitness_batches = 16
        self._num_generations = 2
        self._num_parents_mating = 10
        self._mutation_percent_genes = 10
        self._spoon_feeding = True

    def _verify_supported_models(self) -> None:
        supported_models = (Llama2Provider, OpenAIProvider)
        for llm in self._llms:
            if not isinstance(llm, supported_models):
                raise ValueError(f"Genetic attack mode supported LLMs are: {supported_models}. {llm.qualified_model_name} is not supported.")


    def _verify_supported_classifiers(self) -> None:
        supported_classifiers = (DisapprovalClassifier, CosineSimilarityClassifier)
        if not self._classifiers:
            raise ValueError("No classifiers found, you must provide at least one classifier for this attack mode")

        for classifier in self._classifiers:
            if not isinstance(classifier, supported_classifiers):
                raise ValueError(
                    f"Genetic attack mode supported classifiers are: {supported_classifiers}. {classifier.name} is not supported."
                )

    def _generate_attack_params(self, prompts: list[AdversarialPromptDTO]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for prompt in prompts:
            if SPLIT_TOKEN not in prompt.prompt:
                logger.warning(f"Prompt {prompt.prompt} does not contain the split token '{SPLIT_TOKEN}'")
                continue

            prompt, target = prompt.prompt.split(SPLIT_TOKEN)
            result.append({"prompt": prompt, "target": target})

        return result

    async def _attack(self, prompt: str, target: str, **extra: Any) -> Optional[AttackResultEntry]:
        result_entry: Optional[AttackResultEntry] = AttackResultEntry(original_prompt=prompt, response="")
        llm: Union[Llama2Provider, OpenAIProvider]

        async with self._borrow_any() as llm:
            tokens_handler: TokensHandler = llm.tokens_handler
            new_enc_suffixes, _ = tokens_handler.generate_suffixes(seed=self._population_seed, batch_size=self._suffixes_batch_size)
            initial_population = new_enc_suffixes.to("cpu").numpy()
            gene_space = range(tokens_handler.vocabulary_size)  # Assuming token IDs start from 0
            gene_type = int  # Token IDs are integers

            extra_params = {}
            if self._spoon_feeding:
                llm.add_to_history([BaseLLMProviderResponse(prompt=prompt, response=target)])
                extra_params[LLMProviderExtraParams.APPEND_LAST_RESPONSE] = True
                extra_params[LLMProviderExtraParams.MAX_LENGTH] = 120

            response_loss_cache = {}
            response_clf_cache = {}
            loss_response_cache = {}
            loss_clf_cache = {}

            self.best_solution = None
            self.best_solution_fitness = -1  # Assuming fitness is always >= 0

            def fitness_func(ga_instance, solutions, solution_idx):
                generation = ga_instance.generations_completed
                decoded_solutions = tokens_handler.batch_decode(solutions)

                logger.debug(f"Gen: {generation}, {solution_idx[0]}-{solution_idx[-1]}")
                logger.debug(f"Prompt: {prompt} | Target: {target}")

                scores = []

                for sln_index, decoded_sln in enumerate(decoded_solutions):
                    logger.debug(f"Gen: {generation}, Sln {sln_index}/{solution_idx[-1]}: {decoded_sln}")
                    logger.debug(f"Prompt: {prompt} | Target: {target}")
                    prompt_and_suffix = f"{prompt} {decoded_sln}"

                    model_response: str = llm.sync_generate(prompt_and_suffix, **extra_params).response  # , include_prompt=True)

                    if model_response in response_loss_cache:
                        loss = response_loss_cache[model_response]
                        zipped_clfs_results = response_clf_cache[model_response]
                    else:
                        sanitized_response = model_response.replace("\n\n", " ").replace("\n", " ").replace("\\", "").replace("*", "")

                        clf_scores = []
                        clf_results = []
                        for classifier in self._classifiers:
                            clf_result, score = classifier.sync_classify(sanitized_response, target)
                            clf_scores.append(score)
                            clf_results.append(clf_result)

                        loss: float = 1.0 / np.mean(clf_scores)
                        response_loss_cache[model_response] = loss
                        loss_response_cache[loss] = model_response
                        zipped_clfs_results = {clf.name: clf_result for clf, clf_result in zip(self._classifiers, clf_results)}
                        response_clf_cache[model_response] = zipped_clfs_results
                        loss_clf_cache[loss] = zipped_clfs_results

                    logger.debug(f"Gen: {generation}, Sln {sln_index}/{solution_idx[-1]}: Loss: {str(loss)}, Response: {model_response}")
                    scores.append(loss)

                return scores

            def on_generation(ga_instance):
                # this supports Gen 0
                generation = ga_instance.generations_completed
                prev_best_solution, prev_best_fitness, _ = ga_instance.best_solution(pop_fitness=ga_instance.previous_generation_fitness)
                current_best_solution, current_best_fitness, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
                logger.debug(f"Generation          = {ga_instance.generations_completed}")
                logger.debug(f"Gen Best Solution   = {current_best_solution}")
                logger.debug(f"Gen Best Fitness    = {current_best_fitness}")

                # This part handles the comparison between gen 1 with gen 0 because on_generation doesn't fire between gen 0 and gen 1
                # when gen > 1 previous_generation_fitness == last_generation_fitness
                if generation == 1 and self.best_solution_fitness == -1:
                    if prev_best_fitness > current_best_fitness:
                        current_best_solution = prev_best_solution
                        current_best_fitness = prev_best_fitness

                if current_best_fitness > self.best_solution_fitness:
                    self.best_solution = current_best_solution
                    self.best_solution_fitness = current_best_fitness

                logger.debug(f"Best Solution       = {self.best_solution}")
                logger.debug(f"Best Fitness        = {self.best_solution_fitness}")
                logger.debug(f"Best Response: {loss_response_cache[self.best_solution_fitness]}")

            ga_instance = pygad.GA(
                num_generations=self._num_generations,
                num_parents_mating=self._num_parents_mating,
                fitness_func=fitness_func,
                on_generation=on_generation,
                fitness_batch_size=self._fitness_batches,
                gene_type=gene_type,
                gene_space=gene_space,
                initial_population=initial_population,
                mutation_by_replacement=True,
                mutation_type="random",
                mutation_percent_genes=self._mutation_percent_genes,
            )

            ga_instance.run()
            llm.clear_history()

            # Decode the best solution
            best_solution_decoded = tokens_handler.decode(self.best_solution)
            best_prompt = f"{prompt} {best_solution_decoded}"
            best_loss_response = loss_response_cache[self.best_solution_fitness]
            best_loss_clf = loss_clf_cache[self.best_solution_fitness]

            logger.debug(f"Best Solution: {best_solution_decoded}")
            logger.debug(f"Best Loss: {self.best_solution_fitness}")
            logger.debug(f"Best Response: {best_loss_response}")

            extra_data = {
                "target": target,
                "best_solution": best_solution_decoded,
                "best_loss": self.best_solution_fitness,
                "population_seed": self._population_seed,
                "suffixes_batch_size": self._suffixes_batch_size,
                "fitness_batches": self._fitness_batches,
                "num_generations": self._num_generations,
                "num_parents_mating": self._num_parents_mating,
                "mutation_percent_genes": self._mutation_percent_genes,
            }

            result_entry.current_prompt = best_prompt
            result_entry.response = best_loss_response
            result_entry.classifications = best_loss_clf
            result_entry.extra = extra_data

            return result_entry
