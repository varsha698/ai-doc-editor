from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List

import services.vector_service as vector_service

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingIndexTask:
    document_id: str
    text: str
    text_hash: str


_queue: asyncio.Queue[EmbeddingIndexTask] | None = None
_worker_task: asyncio.Task[None] | None = None
_loop: asyncio.AbstractEventLoop | None = None
_stopping: asyncio.Event | None = None

_BATCH_WINDOW_MS = int(os.getenv("VECTOR_INDEX_BATCH_WINDOW_MS", "250"))
_MAX_BATCH_TASKS = int(os.getenv("VECTOR_INDEX_MAX_BATCH_TASKS", "12"))
_MAX_BATCH_CHUNKS = int(os.getenv("VECTOR_INDEX_MAX_BATCH_CHUNKS", "600"))


async def _embed_and_store_batch(tasks: List[EmbeddingIndexTask]) -> None:
    # Build chunk lists (fast string slicing).
    processed_tasks: List[EmbeddingIndexTask] = []
    doc_chunks: List[List[str]] = []
    flat_chunks: List[str] = []
    for t in tasks:
        if vector_service.is_document_indexed(t.document_id, t.text_hash):
            continue
        chunks = vector_service.chunk_document_text(t.text)
        if not chunks:
            continue
        processed_tasks.append(t)
        doc_chunks.append(chunks)
        flat_chunks.extend(chunks)

    if not flat_chunks:
        return

    # Embed everything in one call (true request batching).
    embeddings = await asyncio.to_thread(vector_service.embed_document_chunks, flat_chunks)

    # Partition embeddings back into per-document chunks and store.
    cursor = 0
    for t, chunks in zip(processed_tasks, doc_chunks):
        if vector_service.is_document_indexed(t.document_id, t.text_hash):
            # If it became indexed while we were embedding, skip storing.
            cursor += len(chunks)
            continue
        n = len(chunks)
        part = embeddings[cursor : cursor + n]
        cursor += n
        vector_service.store_embeddings(t.document_id, chunks, part)
        vector_service.set_document_indexed(t.document_id, t.text_hash, n)


async def _worker_loop() -> None:
    assert _queue is not None
    assert _stopping is not None

    while not _stopping.is_set():
        try:
            first = await asyncio.wait_for(_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            continue

        tasks: List[EmbeddingIndexTask] = [first]
        batch_start = asyncio.get_running_loop().time()

        # Drain more tasks within a small time window for batching.
        while len(tasks) < _MAX_BATCH_TASKS:
            elapsed_ms = (asyncio.get_running_loop().time() - batch_start) * 1000.0
            if elapsed_ms >= _BATCH_WINDOW_MS:
                break
            try:
                nxt = _queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            tasks.append(nxt)

        # Enforce max chunks in batch by truncating tasks.
        # If we can't include all dequeued tasks, we re-queue the remainder.
        total_chunks = 0
        trimmed: List[EmbeddingIndexTask] = []
        requeue: List[EmbeddingIndexTask] = []
        for t in tasks:
            if vector_service.is_document_indexed(t.document_id, t.text_hash):
                # Already indexed; no need to process.
                continue
            est = len(vector_service.chunk_document_text(t.text))
            if est <= 0:
                continue
            if total_chunks + est > _MAX_BATCH_CHUNKS and trimmed:
                requeue.append(t)
                continue
            total_chunks += est
            trimmed.append(t)

        if trimmed:
            try:
                await _embed_and_store_batch(trimmed)
            except Exception:
                LOGGER.exception("Embedding batch failed")

        # Put any remaining tasks back so they are not dropped.
        for t in requeue:
            try:
                _queue.put_nowait(t)
            except Exception:
                LOGGER.exception("Failed re-queueing indexing task")

        # Mark dequeued tasks as done.
        for _ in tasks:
            _queue.task_done()


def start_embedding_indexer(num_workers: int = 1) -> None:
    """
    Start the embedding indexer background worker.

    Note: this should be called from the FastAPI startup event where an event loop exists.
    """
    global _queue, _worker_task, _loop, _stopping
    if _queue is not None:
        return

    _loop = asyncio.get_running_loop()
    _queue = asyncio.Queue()
    _stopping = asyncio.Event()

    # Single worker with batching; additional workers can reduce queue latency but increase GPU pressure.
    if num_workers != 1:
        LOGGER.warning("Only num_workers=1 is supported for now; ignoring %s.", num_workers)
    _worker_task = asyncio.create_task(_worker_loop())
    LOGGER.info("Embedding indexer worker started (batching window=%sms).", _BATCH_WINDOW_MS)


def stop_embedding_indexer() -> None:
    global _worker_task, _queue, _loop, _stopping
    if _queue is None or _stopping is None:
        return
    assert _loop is not None
    assert _stopping is not None

    def _stop() -> None:
        _stopping.set()

    _loop.call_soon_threadsafe(_stop)
    _worker_task = None
    _queue = None
    _loop = None
    _stopping = None


def enqueue_indexing(document_id: str, text: str) -> None:
    """Thread-safe fire-and-forget enqueue."""
    global _queue
    if _queue is None:
        # If the app hasn't started the worker, fall back to synchronous indexing.
        try:
            text_hash = vector_service.compute_document_hash(text)
            if vector_service.is_document_indexed(document_id, text_hash):
                return
            chunks = vector_service.chunk_document_text(text)
            if not chunks:
                return
            embeddings = vector_service.embed_document_chunks(chunks)
            vector_service.store_embeddings(document_id, chunks, embeddings)
            vector_service.set_document_indexed(document_id, text_hash, len(chunks))
        except Exception:
            LOGGER.exception("enqueue_indexing fallback sync indexing failed")
        return

    text_hash = vector_service.compute_document_hash(text)
    task = EmbeddingIndexTask(document_id=document_id, text=text, text_hash=text_hash)

    assert _loop is not None
    asyncio.run_coroutine_threadsafe(_queue.put(task), _loop)

