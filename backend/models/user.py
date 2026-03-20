from pydantic import BaseModel


class StyleProfile(BaseModel):
    avg_sentence_length: float
    tone: str
    vocabulary_complexity: float


class User(BaseModel):
    id: str
    email: str
    style_profile: StyleProfile | None = None
