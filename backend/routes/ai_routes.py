from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.ai_service import analyze_document, chat_edit, chat_edit_stream
from services.suggestion_engine import local_suggestions

router = APIRouter()
LOGGER = logging.getLogger(__name__)

_analysis_cache: dict[str, tuple[dict, datetime]] = {}
_CACHE_TTL_SECONDS = 45


class AnalyzeRequest(BaseModel):
    document_id: str
    text: str
    model: str | None = None


class ChatRequest(BaseModel):
    document_id: str
    message: str
    text: str
    command: str | None = None
    model: str | None = None


def _cache_key(document_id: str, text: str, model: str | None) -> str:
    return f"{document_id}:{hash(text)}:{model or 'default'}"


@router.post("/analyze")
async def analyze(payload: AnalyzeRequest) -> dict:
    key = _cache_key(payload.document_id, payload.text, payload.model)
    now = datetime.utcnow()
    cached = _analysis_cache.get(key)
    if cached and now - cached[1] < timedelta(seconds=_CACHE_TTL_SECONDS):
        return cached[0]

    try:
        result = await analyze_document(payload.document_id, payload.text, payload.model)
    except Exception:
        LOGGER.exception("/ai/analyze failed")
        raise
    _analysis_cache[key] = (result, now)
    return result


@router.post("/chat")
async def chat(payload: ChatRequest) -> dict:
    try:
        return await chat_edit(
            payload.document_id, payload.message, payload.text, payload.command, payload.model
        )
    except Exception:
        LOGGER.exception("/ai/chat failed")
        raise


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    async def event_gen():
        try:
            async for token in chat_edit_stream(
                payload.document_id,
                payload.message,
                payload.text,
                payload.command,
                payload.model,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            LOGGER.exception("/ai/chat/stream failed")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.websocket("/chat/ws")
async def chat_ws(websocket: WebSocket) -> None:
    """
    WebSocket streaming endpoint for /ai/chat.

    Client sends: { document_id, message, text, command? }
    Server sends incremental frames:
      - { type: "suggestions", suggestions: Suggestion[] }
      - { type: "token", token: "..." }
      - { type: "done" }
      - { type: "error", error: "..." }
    """
    await websocket.accept()
    try:
        raw = await websocket.receive_json()
        payload = ChatRequest(**raw)

        # Suggestions are local / deterministic; send them immediately.
        suggestions = local_suggestions(payload.text)
        await websocket.send_json({"type": "suggestions", "suggestions": suggestions})

        async for token in chat_edit_stream(
            payload.document_id, payload.message, payload.text, payload.command, payload.model
        ):
            await websocket.send_json({"type": "token", "token": token})

        # Final suggestions frame (same as initial, but keeps the protocol explicit).
        suggestions_final = local_suggestions(payload.text)
        await websocket.send_json({"type": "suggestions", "suggestions": suggestions_final})

        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected during /ai/chat/ws")
    except Exception as e:
        LOGGER.exception("/ai/chat/ws failed")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
