from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Dict

from services.grammar_service import analyze_grammar, issues_to_suggestions
from services.suggestion_engine import build_scores, local_suggestions
from services.style_engine import analyze_user_style, apply_style_constraints
from services.model_manager import (
    get_active_local_llm,
    get_active_model_key,
    get_generation_semaphore,
)
from services.vector_service import retrieve_relevant_context, schedule_document_indexing

LOGGER = logging.getLogger(__name__)


def _rag_prefix(retrieved: str) -> str:
    if not retrieved.strip():
        return ""
    return (
        "Retrieved relevant excerpts from the document (vector search):\n\n"
        f"{retrieved}\n\n"
        "---\n\n"
    )


def _format_llm_with_rag(retrieved: str, body: str, max_body: int = 6000) -> str:
    prefix = _rag_prefix(retrieved)
    trimmed = body[:max_body] if body else ""
    if not prefix:
        return trimmed
    return f"{prefix}Full document text (may be truncated):\n{trimmed}"


async def _augment_with_rag(
    document_id: str, query_hint: str, body: str, max_body: int = 6000
) -> str:
    ctx = await asyncio.to_thread(retrieve_relevant_context, document_id, query_hint)
    return _format_llm_with_rag(ctx, body, max_body)


def _user_docs_for_document(document_id: str, fallback_text: str) -> tuple[str, list[str]]:
    # Uses the in-memory document store already maintained by document routes.
    try:
        from routes.document_routes import _documents  # type: ignore
    except Exception:
        return document_id, [fallback_text]
    doc = _documents.get(document_id) or {}
    user_id = doc.get("user_id") or document_id
    related = [
        str(item.get("content", "")).strip()
        for item in _documents.values()
        if item.get("user_id") == user_id and str(item.get("content", "")).strip()
    ]
    if not related and fallback_text.strip():
        related = [fallback_text]
    return str(user_id), related


def _chat_task(command: str | None) -> str:
    if command == "summarize":
        return "summary"
    if command == "create-outline":
        return "outline"
    if command == "make-professional":
        return "style"
    return "rewrite"


def _attach_ranges_to_suggestions(text: str, suggestions: list[Dict]) -> list[Dict]:
    """Best-effort start/end assignment for suggestions missing ranges.

    The frontend only applies edits when it can map and the extracted span matches
    `original_text`, so attaching ranges via exact substring match is safe.
    """
    if not text:
        return suggestions
    for s in suggestions:
        if s.get("start") is not None and s.get("end") is not None:
            continue
        original = s.get("original_text")
        if not original or not str(original).strip():
            continue
        original_str = str(original)
        idx = text.find(original_str)
        if idx < 0:
            continue
        s["start"] = idx
        s["end"] = idx + len(original_str)
    return suggestions


async def analyze_document(document_id: str, text: str, model: str | None = None) -> Dict:
    try:
        await asyncio.to_thread(schedule_document_indexing, document_id, text)
    except Exception:
        LOGGER.exception("analyze_document: vector index scheduling failed")

    snippet = text[:8000]
    try:
        grammar_result = await asyncio.to_thread(analyze_grammar, snippet)
    except Exception:
        LOGGER.exception("analyze_document: grammar_service failed")
        grammar_result = {
            "issues": [],
            "readability_score": 55,
            "clarity_score": 55,
            "grammar_score": 75,
        }

    grammar_issues = grammar_result.get("issues", [])
    grammar_suggestions = issues_to_suggestions(snippet, grammar_issues[:35])

    rag_query = text.strip()[:2500] or text.strip()
    llm_input = await _augment_with_rag(document_id, rag_query, text, max_body=6000)
    user_id, user_docs = _user_docs_for_document(document_id, text)
    style_profile = await asyncio.to_thread(analyze_user_style, user_id, user_docs)
    llm_input = apply_style_constraints(llm_input, style_profile)

    llm = get_active_local_llm(model)
    model_key = get_active_model_key(model)
    sem = get_generation_semaphore(model_key)
    llm_summary = ""
    model_suggestions: list = []
    try:
        await sem.acquire()
        try:
            llm_summary = await llm.generate("summary", llm_input)
        finally:
            sem.release()
    except Exception:
        LOGGER.exception("analyze_document: summary generation failed")
    try:
        await sem.acquire()
        try:
            model_suggestions = await llm.suggest(llm_input)
        finally:
            sem.release()
    except Exception:
        LOGGER.exception("analyze_document: suggestion generation failed")
        model_suggestions = []

    suggestions = grammar_suggestions + model_suggestions + local_suggestions(text)
    suggestions = _attach_ranges_to_suggestions(text, suggestions)
    scores = build_scores(text)
    scores["readability"] = int(
        0.35 * scores["readability"] + 0.65 * grammar_result["readability_score"]
    )
    scores["clarity"] = int(0.35 * scores["clarity"] + 0.65 * grammar_result["clarity_score"])
    scores["grammar"] = grammar_result["grammar_score"]

    summary = llm_summary[:400] if llm_summary else "Analysis complete."
    return {
        "suggestions": suggestions,
        "scores": scores,
        "summary": summary,
        "grammar_issues": grammar_issues,
        "style_profile": style_profile,
    }


async def chat_edit(
    document_id: str,
    message: str,
    text: str,
    command: str | None = None,
    model: str | None = None,
) -> Dict:
    try:
        await asyncio.to_thread(schedule_document_indexing, document_id, text)
    except Exception:
        LOGGER.exception("chat_edit: vector index scheduling failed")

    rag_query = f"{message}\n\n{text.strip()[:1200]}"
    doc_block = await _augment_with_rag(document_id, rag_query, text, max_body=6000)
    user_id, user_docs = _user_docs_for_document(document_id, text)
    style_profile = await asyncio.to_thread(analyze_user_style, user_id, user_docs)
    doc_block = apply_style_constraints(doc_block, style_profile)

    llm = get_active_local_llm(model)
    model_key = get_active_model_key(model)
    sem = get_generation_semaphore(model_key)
    task = _chat_task(command)
    payload = f"Instruction: {message}\n\n{doc_block}"
    try:
        await sem.acquire()
        try:
            reply = await llm.generate(task, payload)
        finally:
            sem.release()
    except Exception:
        LOGGER.exception("chat_edit: generation failed")
        reply = ""
    suggestions = local_suggestions(text)
    return {
        "reply": reply or "I reviewed your text and suggest improving clarity and structure.",
        "suggestions": suggestions,
    }


async def chat_edit_stream(
    document_id: str,
    message: str,
    text: str,
    command: str | None = None,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    try:
        await asyncio.to_thread(schedule_document_indexing, document_id, text)
    except Exception:
        LOGGER.exception("chat_edit_stream: vector index scheduling failed")

    rag_query = f"{message}\n\n{text.strip()[:1200]}"
    doc_block = await _augment_with_rag(document_id, rag_query, text, max_body=6000)
    user_id, user_docs = _user_docs_for_document(document_id, text)
    style_profile = await asyncio.to_thread(analyze_user_style, user_id, user_docs)
    doc_block = apply_style_constraints(doc_block, style_profile)

    llm = get_active_local_llm(model)
    model_key = get_active_model_key(model)
    sem = get_generation_semaphore(model_key)
    task = _chat_task(command)
    payload = f"Instruction: {message}\n\n{doc_block}"
    try:
        await sem.acquire()
        try:
            async for token in llm.stream_generate(task, payload):
                yield token
        finally:
            sem.release()
    except Exception:
        LOGGER.exception("chat_edit_stream failed")
        raise
