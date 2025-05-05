import asyncio
import logging
import re
from string import ascii_uppercase
from typing import Any, Optional, Type

from art import text2art
from pydantic import BaseModel, Field

from fuzzyai.handlers.attacks.artprompt.prompts import JAILBREAK_PROMPT, JAILBREAK_SIMPLE_PROMPT, LEARN_FONT_PROMPT
from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.db.adv_prompts import AdversarialPromptDTO
from fuzzyai.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class ArtPromptAttackHandlerExtraParams(BaseModel):
    blacklisted_words: Optional[str] = Field(None, description="comma-separated words to mask in the attack. Could be either a word or a phrase.")
    simple_mode: bool = Field(True, description="use simple mode for the attack (i.e hash separated letters as ascii art). Default: True")
    disable_font_learn: bool = Field(True, description="disable font learning. Default: True")
    
@attack_handler_fm.flavor(FuzzerAttackMode.ARTPROMPT)
class ArtPromptAttackHandler(BaseAttackTechniqueHandler[ArtPromptAttackHandlerExtraParams]):
    """
    Attack handler for ArtPrompt attack technique (https://arxiv.org/abs/2402.11753)
    """

    def __init__(self, **extra: Any):
        super().__init__(**extra)

        self._font_map: dict[str, str] = {}
        self._font_learn_task: Optional[asyncio.Task[None]] = None

        self._blacklisted_words: list[str] = self._extra_args.blacklisted_words.split(",") if self._extra_args.blacklisted_words else []
        self._simple_mode = self._extra_args.simple_mode

        if self._simple_mode:
            self._jailbreak_prompt = JAILBREAK_SIMPLE_PROMPT
            self._font_learn_task = asyncio.create_task(asyncio.sleep(0))
        else:
            self._jailbreak_prompt = JAILBREAK_PROMPT
            if self._extra_args.disable_font_learn:
                logger.debug("Font learning disabled, generating local font map...")
                self._font_learn_task = asyncio.create_task(self._generate_font_map())

    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return ArtPromptAttackHandlerExtraParams

    async def _generate_font_map(self) -> None:
        for let in ascii_uppercase:
            self._font_map[let] = text2art(let, "blocks") 
            logger.debug("Generated font for letter %s:\n%s", let, self._font_map[let])

    async def _learn_font(self) -> None:
        logger.debug("Bruteforcing fonts...")
        logger.warning("Pay attention: this might not work with the original attack prompt")

        async with self._borrow(self._model) as llm:
            for let in ascii_uppercase:
                prompt = LEARN_FONT_PROMPT.format(let)
                response = await llm.generate(prompt)
                if response:
                    self._font_map[let] = response.response
                    logger.debug("Learned font for letter %s:\n%s", let, self._font_map[let])
                else:
                    logger.error("Failed to learn font for letter %s", let)
                    raise Exception("ASCII Font learning process failed")

    def _generate_attack_params(self, prompts: list[AdversarialPromptDTO]) -> list[dict[str, Any]]:
        return [{"prompt": prompt.prompt} for prompt in prompts]
    
    async def _reduce_attack_params(self, entries: list[AttackResultEntry], 
                                    attack_params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        
        updated_params = [param for param in attack_params
                            if not any(entry.original_prompt == param['prompt']
                                for entry in entries)]
        
        return updated_params
    
    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        result: Optional[AttackResultEntry] = None
        llm: BaseLLMProvider
        classifications: dict[str, Any] = {}
        banner: str = str()
        harmful_word: Optional[str] = None
        masked_prompt = str(prompt)

        if self._font_learn_task is None:
            self._font_learn_task = asyncio.create_task(self._learn_font())
        
        await self._font_learn_task

        if self._blacklisted_words:
            if len([word for word in self._blacklisted_words if word.strip() in prompt]) > 1:
                logger.warning("More than one harmful word/phrase found in prompt: %s", prompt)
            
            for word in self._blacklisted_words:
                if word in prompt:
                    harmful_word = word
                    break

        # If not blacklisted words found, try to find [masked words]
        if not harmful_word:
            found_words: list[str] = self._extract_harmful_words(prompt)
            if not found_words:
                logger.error("No [masked words] found in prompt: %s", prompt)
                return None

            harmful_word = found_words[0]

        masked_prompt = masked_prompt.replace(f"[{harmful_word}]", "[MASK]").replace(f"{harmful_word}", "[MASK]")
        banner = self._create_banner_simple(harmful_word) if self._simple_mode else self._create_banner(harmful_word)
        adv_prompt = self._jailbreak_prompt.format(harmful_word_length=len(harmful_word), banner=banner, masked_prompt=masked_prompt)

        async with self._borrow(self._model) as llm:
            logger.debug("Generating response for prompt: %s", adv_prompt)
            response = await llm.generate(adv_prompt, **self._extra)
            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt=adv_prompt, 
                                       response=response.response) if response else None
            logger.debug("Response: %s", response.response if response else "None")
            
        classifications = await self._classify_llm_response(response, original_prompt=prompt)
        
        if result:
            result.classifications = classifications

        return result

    def _extract_harmful_words(self, prompt: str) -> list[str]:
        # Extract the harmful words. It can either be a single word (i.e "hello [bomb]") or a phrase (i.e "hello [bomb nuke] world")
        matches = re.findall(r'\[([^\]]+)\]', prompt)
        return [match for match in matches]
    
    def _create_banner_simple(self, word: str) -> str:
        border = (len(word) + len(word) + 1) * "#"
        return f"{border}\n#{'#'.join([l.upper() for l in word])}#\n{border}"
    
    def _create_banner(self, word: str) -> str:
        # Iterate through each character in the word
        lines = []
        for char_idx, char in enumerate(word):
            # Calculate ASCII art for the character
            char_art = text2art(char, "blocks")
            # Split the ASCII art into lines
            char_lines = char_art.split('\n')
            char_lines.remove('')
            if char_idx < len(word) - 1:
                for i,l in enumerate(char_lines):
                    char_lines[i] = l + "*"
                
            # Append the lines to the list of lines
            lines.append(char_lines)

        # Transpose the list of lines
        combined_lines = zip(*lines)
        # Print the combined ASCII art
        output: list[str] = []
        for line in combined_lines:
            output.append("".join(line))
        result = "\n".join(output)
        return result.replace((len(word)-1) * "*", '')
    