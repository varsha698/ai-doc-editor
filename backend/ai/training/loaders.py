from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger("ai.training.loaders")


def _trim(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def load_grammar_correction_hf(max_samples: int | None = None, max_input_chars: int = 2000) -> List[Dict[str, Any]]:
    """Load JFLEG-style grammar correction examples (sentence ΓåÆ fluent English)."""
    rows: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset
    except ImportError:
        LOGGER.error("datasets package required for HF loaders.")
        return rows

    try:
        ds = load_dataset("jfleg")
    except Exception as e:
        try:
            ds = load_dataset("jfleg", trust_remote_code=True)
        except Exception as e2:
            LOGGER.warning("Could not load HF dataset 'jfleg': %s / %s", e, e2)
            return rows

    split = ds.get("validation") or ds.get("test") or ds.get("train")
    if split is None:
        return rows

    limit = max_samples if max_samples is not None else len(split)
    for i, ex in enumerate(split):
        if i >= limit:
            break
        sentence = ex.get("sentence")
        corrections = ex.get("corrections")
        if not sentence or not corrections:
            continue
        target = corrections[0] if isinstance(corrections, (list, tuple)) else str(corrections)
        rows.append(
            {
                "task": "grammar",
                "input_text": _trim(str(sentence), max_input_chars),
                "instruction": "Correct the grammar and fluency. Preserve the original meaning.",
                "output_text": _trim(str(target), max_input_chars),
                "metadata": {"source": "hf:jfleg", "idx": i},
            }
        )
    LOGGER.info("grammar (jfleg): loaded %d rows", len(rows))
    return rows


def load_summarization_hf(max_samples: int | None = None, max_input_chars: int = 6000) -> List[Dict[str, Any]]:
    """Load CNN/DailyMail-style summarization (article ΓåÆ highlights)."""
    rows: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset
    except ImportError:
        return rows

    try:
        ds = load_dataset("cnn_dailymail", "3.0.0")
    except Exception as e:
        LOGGER.warning("Could not load HF dataset 'cnn_dailymail': %s", e)
        return rows

    split = ds.get("train") or ds["train"]
    limit = max_samples if max_samples is not None else len(split)
    for i, ex in enumerate(split):
        if i >= limit:
            break
        article = ex.get("article") or ""
        highlights = ex.get("highlights") or ""
        if not article or not highlights:
            continue
        rows.append(
            {
                "task": "summarize",
                "input_text": _trim(str(article), max_input_chars),
                "instruction": "Summarize the article in concise bullet points.",
                "output_text": _trim(str(highlights), max_input_chars // 2),
                "metadata": {"source": "hf:cnn_dailymail", "idx": i},
            }
        )
    LOGGER.info("summarization (cnn_dailymail): loaded %d rows", len(rows))
    return rows


def load_rewrite_paraphrase_hf(max_samples: int | None = None, max_input_chars: int = 1500) -> List[Dict[str, Any]]:
    """Load PAWS paraphrase pairs as rewrite tasks (sentence1 ΓåÆ sentence2)."""
    rows: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset
    except ImportError:
        return rows

    try:
        ds = load_dataset("paws", "labeled_final")
    except Exception as e:
        LOGGER.warning("Could not load HF dataset 'paws': %s", e)
        return rows

    split = ds.get("train") or ds["train"]
    limit = max_samples if max_samples is not None else len(split)
    for i, ex in enumerate(split):
        if i >= limit:
            break
        if ex.get("label") != 1:
            continue
        s1, s2 = ex.get("sentence1"), ex.get("sentence2")
        if not s1 or not s2:
            continue
        rows.append(
            {
                "task": "rewrite",
                "input_text": _trim(str(s1), max_input_chars),
                "instruction": "Rewrite the sentence for clarity and flow while preserving meaning.",
                "output_text": _trim(str(s2), max_input_chars),
                "metadata": {"source": "hf:paws", "idx": i},
            }
        )
    LOGGER.info("rewrite (paws labeled_final): loaded %d rows", len(rows))
    return rows


def load_user_docs_local(
    user_docs_dir: Path,
    max_chunk_chars: int = 1800,
    overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Build synthetic instruction rows from plain .txt samples (three task types per chunk)."""
    rows: List[Dict[str, Any]] = []
    if not user_docs_dir.is_dir():
        LOGGER.warning("User docs directory missing: %s", user_docs_dir)
        return rows

    paths = sorted(user_docs_dir.rglob("*.txt"))
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        for para in paragraphs:
            if len(para) <= max_chunk_chars:
                chunks.append(para)
            else:
                start = 0
                while start < len(para):
                    end = min(start + max_chunk_chars, len(para))
                    piece = para[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    if end == len(para):
                        break
                    start = max(0, end - overlap_chars)

        for j, chunk in enumerate(chunks):
            base_meta = {"source": f"local:{path.as_posix()}", "chunk": j}
            rows.extend(
                [
                    {
                        "task": "grammar",
                        "input_text": chunk,
                        "instruction": "Correct grammar and punctuation while preserving voice.",
                        "output_text": chunk,
                        "metadata": {**base_meta, "loader": "user_docs"},
                    },
                    {
                        "task": "rewrite",
                        "input_text": chunk,
                        "instruction": "Rewrite for clarity and stronger flow; keep meaning.",
                        "output_text": chunk,
                        "metadata": {**base_meta, "loader": "user_docs"},
                    },
                    {
                        "task": "summarize",
                        "input_text": chunk,
                        "instruction": "Summarize in 2ΓÇô3 short bullet points.",
                        "output_text": "- Main idea\n- Supporting detail",
                        "metadata": {**base_meta, "loader": "user_docs"},
                    },
                ]
            )
    LOGGER.info("user_docs: generated %d rows from %d files", len(rows), len(paths))
    return rows
