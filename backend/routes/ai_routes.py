from __future__ import annotations

from datetime import datetime, timedelta
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from services.ai_service import analyze_document, chat_edit
from services.embedding_service import embed_texts

router = APIRouter()

_analysis_cache: dict[str, tuple[dict, datetime]] = {}
_CACHE_TTL_SECONDS = 45


class AnalyzeRequest(BaseModel):
    document_id: str
    text: str


class ChatRequest(BaseModel):
    document_id: str
    message: str
    text: str
    command: str | None = None


def _cache_key(document_id: str, text: str) -> str:
    return f"{document_id}:{hash(text)}"


def _background_embed(document_id: str, text: str) -> None:
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        return
    # Embeddings are generated asynchronously and can be persisted to vector DB.
    _ = embed_texts(chunks[:40])


@router.post("/analyze")
async def analyze(payload: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict:
    key = _cache_key(payload.document_id, payload.text)
    now = datetime.utcnow()
    cached = _analysis_cache.get(key)
    if cached and now - cached[1] < timedelta(seconds=_CACHE_TTL_SECONDS):
        return cached[0]

    result = await analyze_document(payload.text)
    _analysis_cache[key] = (result, now)
    background_tasks.add_task(_background_embed, payload.document_id, payload.text)
    return result


@router.post("/chat")
async def chat(payload: ChatRequest) -> dict:
    return await chat_edit(payload.message, payload.text, payload.command)
