import logging
from typing import Any, Optional

from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.base import llm_provider_fm
from fuzzyai.llm.providers.enums import LLMProvider, LLMProviderExtraParams
from fuzzyai.llm.providers.local.local_llm import LocalProvider, LocalProviderException

logger = logging.getLogger(__name__)


class Llama2ProviderException(LocalProviderException):
    pass


@llm_provider_fm.flavor(LLMProvider.LOCAL_LLAMA2)
class Llama2Provider(LocalProvider):
    # <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]
    CONV_USER_AND_HISTORY_FORMAT = "<s>[INST] {user_message} [/INST] {model_response} "
    CONV_USER_FORMAT = "<s>[INST] {user_message} [/INST] "

    def __init__(self, model: str, tokenizer_path: Optional[str] = None, device: str = "cuda:0", **kwargs: Any) -> None:
        super().__init__(model, tokenizer_path, device, **kwargs)

        self._tokenizer.pad_token = self._tokenizer.unk_token
        self._tokenizer.padding_side = "left"

    def sync_generate(self, prompt: str, append_last_response: Optional[bool] = None, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            if append_last_response:
                if history := self.get_history():
                    prompt = self.CONV_USER_AND_HISTORY_FORMAT.format(user_message=prompt, model_response=history[-1].response)
            else:
                prompt = self.CONV_USER_FORMAT.format(user_message=prompt)

            """Generate text synchronously using the model."""
            generate_text_extra_params = {k:v for k, v in extra.items() if k not in [LLMProviderExtraParams.APPEND_LAST_RESPONSE]}
            response = self._text_gen_handler.generate_text(prompt, **generate_text_extra_params)
            return BaseLLMProviderResponse(response=response)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise Llama2ProviderException("Cant generate text")
