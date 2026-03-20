from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_documents: dict[str, dict] = {}


class DocumentPayload(BaseModel):
    id: str
    user_id: str
    title: str
    content: str


@router.post("/")
async def upsert_document(payload: DocumentPayload) -> dict:
    _documents[payload.id] = payload.model_dump()
    return {"ok": True, "document": _documents[payload.id]}


@router.get("/{document_id}")
async def get_document(document_id: str) -> dict:
    return {"document": _documents.get(document_id)}
