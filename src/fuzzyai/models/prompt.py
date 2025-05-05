from pydantic import BaseModel


class BasePrompt(BaseModel):
    prompt: str
