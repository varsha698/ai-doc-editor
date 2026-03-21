from __future__ import annotations

import logging
import os
import re
import threading
import hashlib
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import List

import chromadb

from services.embedding_service import embed_texts

LOGGER = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CHROMA_DIR = _BACKEND_DIR / "chroma_data"

_chunk_chars = int(os.getenv("VECTOR_CHUNK_CHARS", "900"))
_chunk_overlap = int(os.getenv("VECTOR_CHUNK_OVERLAP", "120"))
_default_top_k = int(os.getenv("VECTOR_TOP_K", "6"))
_persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(_DEFAULT_CHROMA_DIR))).expanduser().resolve()

_chroma_lock = threading.Lock()

_INDEX_CACHE_DB = _persist_dir / "index_cache.sqlite"
_index_cache_lock = threading.Lock()


def _sanitize_doc_id(document_id: str) -> str:
    return re.sub(r"[^\w\-.:@]+", "_", document_id)[:200] or "default_doc"


def compute_document_hash(text: str) -> str:
    # Hash is used for cache validity checks so we only re-embed when the content changes.
    t = (text or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(t).hexdigest()


def _ensure_index_cache_schema() -> None:
    with _index_cache_lock:
        _persist_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_INDEX_CACHE_DB))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_index_cache (
                    document_id TEXT PRIMARY KEY,
                    text_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


_ensure_index_cache_schema()


def is_document_indexed(document_id: str, text_hash: str) -> bool:
    safe_id = _sanitize_doc_id(document_id)
    with _index_cache_lock:
        conn = sqlite3.connect(str(_INDEX_CACHE_DB))
        try:
            row = conn.execute(
                "SELECT text_hash FROM document_index_cache WHERE document_id = ?",
                (safe_id,),
            ).fetchone()
            if not row:
                return False
            return row[0] == text_hash
        finally:
            conn.close()


def has_document_vectors(document_id: str) -> bool:
    safe_id = _sanitize_doc_id(document_id)
    with _index_cache_lock:
        conn = sqlite3.connect(str(_INDEX_CACHE_DB))
        try:
            row = conn.execute(
                "SELECT chunk_count FROM document_index_cache WHERE document_id = ?",
                (safe_id,),
            ).fetchone()
            return bool(row and int(row[0]) > 0)
        finally:
            conn.close()


def set_document_indexed(document_id: str, text_hash: str, chunk_count: int) -> None:
    safe_id = _sanitize_doc_id(document_id)
    with _index_cache_lock:
        conn = sqlite3.connect(str(_INDEX_CACHE_DB))
        try:
            conn.execute(
                """
                INSERT INTO document_index_cache (document_id, text_hash, chunk_count, updated_at)
                VALUES (?, ?, ?, strftime('%s','now'))
                ON CONFLICT(document_id) DO UPDATE SET
                    text_hash = excluded.text_hash,
                    chunk_count = excluded.chunk_count,
                    updated_at = excluded.updated_at
                """,
                (safe_id, text_hash, int(chunk_count)),
            )
            conn.commit()
        finally:
            conn.close()


@lru_cache(maxsize=1)
def _chroma_client() -> chromadb.PersistentClient:
    _persist_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("ChromaDB persistent path: %s", _persist_dir)
    return chromadb.PersistentClient(path=str(_persist_dir))


def _get_collection():
    client = _chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL, "embedding_dim": str(EMBEDDING_DIM)},
    )


def chunk_document_text(
    text: str,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """Split document into overlapping character windows for embedding."""
    max_chars = max_chars if max_chars is not None else _chunk_chars
    overlap = overlap if overlap is not None else _chunk_overlap
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def embed_document_chunks(chunks: List[str]) -> List[List[float]]:
    """Encode chunk texts with the same model as query-time retrieval."""
    if not chunks:
        return []
    try:
        return embed_texts(chunks)
    except Exception:
        LOGGER.exception("embed_document_chunks failed")
        raise


def store_embeddings(
    document_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
) -> None:
    """
    Replace all stored chunks for ``document_id`` with the new vectors.
    """
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        LOGGER.warning("store_embeddings: skip empty or length mismatch for doc %s", document_id)
        return

    safe_id = _sanitize_doc_id(document_id)
    ids = [f"{safe_id}:chunk:{i}" for i in range(len(chunks))]
    metadatas = [{"document_id": safe_id, "chunk_index": i} for i in range(len(chunks))]

    try:
        with _chroma_lock:
            col = _get_collection()
            col.delete(where={"document_id": safe_id})
            col.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
        LOGGER.info("Stored %d chunks for document_id=%s", len(chunks), safe_id)
    except Exception:
        LOGGER.exception("store_embeddings failed for %s", document_id)
        raise


def retrieve_relevant_context(
    document_id: str,
    query_text: str,
    top_k: int | None = None,
) -> str:
    """
    Query Chroma for chunks most similar to ``query_text`` for this document.
    Returns a single string suitable to prepend to LLM prompts.
    """
    if not query_text or not query_text.strip():
        return ""

    # If this document hasn't been indexed yet, skip Chroma querying.
    if not has_document_vectors(document_id):
        return ""

    top_k = top_k if top_k is not None else _default_top_k
    safe_id = _sanitize_doc_id(document_id)

    try:
        q_emb = embed_texts([query_text.strip()[:2000]])[0]
    except Exception:
        LOGGER.exception("retrieve_relevant_context: query embed failed")
        return ""

    try:
        with _chroma_lock:
            col = _get_collection()
            res = col.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                where={"document_id": safe_id},
                include=["documents", "distances"],
            )
    except Exception:
        LOGGER.exception("retrieve_relevant_context: Chroma query failed")
        return ""

    docs = (res.get("documents") or [[]])[0]
    if not docs:
        LOGGER.debug("No vector hits for document_id=%s", safe_id)
        return ""

    parts: List[str] = []
    for i, doc in enumerate(docs):
        if not doc or not str(doc).strip():
            continue
        parts.append(f"[{i + 1}] {str(doc).strip()}")
    return "\n\n".join(parts)


def index_document(document_id: str, text: str) -> None:
    """Chunk, embed, and upsert one document (sync; call via asyncio.to_thread)."""
    chunks = chunk_document_text(text)
    if not chunks:
        return
    embeddings = embed_document_chunks(chunks)
    store_embeddings(document_id, chunks, embeddings)

    text_hash = compute_document_hash(text)
    set_document_indexed(document_id, text_hash, len(chunks))


def schedule_document_indexing(document_id: str, text: str) -> None:
    """Enqueue background indexing if this text hash isn't indexed yet."""
    text_hash = compute_document_hash(text)
    if is_document_indexed(document_id, text_hash):
        return

    # Fire-and-forget background job (falls back to sync if worker is not running).
    from services.embedding_index_queue import enqueue_indexing

    enqueue_indexing(document_id, text)
