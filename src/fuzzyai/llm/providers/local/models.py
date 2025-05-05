from typing import Optional

from pydantic import Field, model_validator

from fuzzyai.consts import PARAMETER_MAX_TOKENS
from fuzzyai.models.base_models import RemoveNoneModel


class LocalGenerateOptions(RemoveNoneModel):
    max_new_tokens: int = Field(50, alias=PARAMETER_MAX_TOKENS) # type: ignore
    temperature: Optional[float] = Field(None)
    top_p: Optional[float] = Field(None)
    do_sample: bool = Field(False)

    @model_validator(mode='after')
    def check_temperature(self) -> 'LocalGenerateOptions':
        if self.temperature is not None or self.top_p is not None:
            self.do_sample = True
        
        return self