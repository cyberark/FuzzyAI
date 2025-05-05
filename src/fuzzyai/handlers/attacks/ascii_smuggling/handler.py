import logging
from typing import Any, Optional, Tuple, Type

from pydantic import BaseModel, Field

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class AsciiSmugglingExtraParams(BaseModel):
    use_unicode_tags: bool = Field(False, description="Add unicode tags in the prompt (START/END)")

@attack_handler_fm.flavor(FuzzerAttackMode.ASCIISMUGGLING)
class AsciiSmugglingAttackHandler(BaseAttackTechniqueHandler[AsciiSmugglingExtraParams]):
    """
    Ascii Smuggling attack handler - converts ascii to unicode chars
    """
    def __init__(self, **extra: Any):
        super().__init__(**extra)

    async def _attack(self, prompt: str, **extra: Any) -> Optional[AttackResultEntry]:
        result: Optional[AttackResultEntry] = None
        llm: BaseLLMProvider
        classifications: dict[str, Any] = {}

        async with self._borrow(self._model) as llm:
            _, encoded_prompt = self._encode_message(prompt, use_unicode_tags=self._extra_args.use_unicode_tags)
            response = await llm.generate(prompt, **self._extra)
            result = AttackResultEntry(original_prompt=prompt,
                                       current_prompt=encoded_prompt, 
                                       response=response.response) if response else None
            logger.debug("Response: %s", response.response if response else "None")
            
        classifications = await self._classify_llm_response(response, original_prompt=prompt)
        
        if result:
            result.classifications = classifications

        return result

    @classmethod
    def extra_args_cls(cls) -> Type[BaseModel]:
        return AsciiSmugglingExtraParams
    
    def _encode_message(self, message: str, use_unicode_tags: bool = False) -> Tuple[str, str]:
        """
        Encodes a message using Unicode code points.
        
        Args:
            message (str): The message to encode
            use_unicode_tags (bool): Whether to add Unicode tags at start and end
            
        Returns:
            Tuple[str, str]: A tuple containing (code_points_str, encoded_str) where:
                - code_points_str is the string of Unicode code points
                - encoded_str is the actual encoded message
        """
        encoded: str = ''
        code_points: str = ''
        invalid_chars: str = ''
        
        # Add starting tag if requested
        if use_unicode_tags:
            encoded += chr(0xE0001)
            code_points += 'U+E0001 '
        
        # Process each character
        for char in message:
            # Check if character is in valid ASCII printable range (0x20 to 0x7E)
            code_point: int = ord(char)
            if 0x20 <= code_point <= 0x7E:
                # Calculate new code point and encode
                new_code_point: int = 0xE0000 + code_point
                encoded += chr(new_code_point)
                code_points += f'U+{new_code_point:X} '
            else:
                invalid_chars += char
        
        # Add ending tag if requested
        if use_unicode_tags:
            encoded += chr(0xE007F)
            code_points += 'U+E007F'
        
        # Instead of updating status directly, we'll print the invalid characters message
        if invalid_chars:
            logger.warning(f'Invalid characters detected: {invalid_chars}')
        
        return code_points.strip(), encoded