from pydantic import BaseModel


class RatingRequest(BaseModel):
    prompt: str
    assistant_response: str

    def __str__(self) -> str:
        return f"[PROMPT]: {self.prompt}\n[ASSISTANT'S RESPONSE]: {self.assistant_response}"
    