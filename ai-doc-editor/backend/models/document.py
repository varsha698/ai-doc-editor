from datetime import datetime
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    user_id: str
    title: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    embedding_model: str
