from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class RemoveNoneModel(BaseModel):
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """
        This Overrides the default model dump method to exclude None values
        """
        return super().model_dump(exclude_none=True)
    
class AliasedBaseModel(RemoveNoneModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )

