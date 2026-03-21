from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Literal, Tuple

from datasets import load_dataset

LOGGER = logging.getLogger(__name__)


TaskName = Literal["grammar", "summarize", "rewrite"]


def _pick_split(ds: Any, candidates: List[str]) -> Any:
    for c in candidates:
        if c in ds:
            return ds[c]
    for c in candidates:
        try:
            return ds[c]
        except Exception:
            continue
    return None


def load_grammar_correction(split: str = "validation", max_samples: int = 200) -> List[Dict[str, str]]:
    """Returns rows: {input_text, output_text}."""
    try:
        ds = load_dataset("jfleg")
    except Exception as e:
        LOGGER.error("Failed to load jfleg: %s", e)
        return []

    chosen = _pick_split(ds, [split, "test", "validation", "train"])
    if chosen is None:
        return []

    rows: List[Dict[str, str]] = []
    for i, ex in enumerate(chosen):
        if i >= max_samples:
            break
        sentence = ex.get("sentence")
        corrections = ex.get("corrections")
        if not sentence or not corrections:
            continue
        target = corrections[0] if isinstance(corrections, (list, tuple)) else str(corrections)
        if not target:
            continue
        rows.append({"input_text": str(sentence), "output_text": str(target)})
    return rows


def load_summarization(split: str = "validation", max_samples: int = 200) -> List[Dict[str, str]]:
    """Returns rows: {input_text, output_text} (article -> highlights)."""
    try:
        ds = load_dataset("cnn_dailymail", "3.0.0")
    except Exception as e:
        LOGGER.error("Failed to load cnn_dailymail: %s", e)
        return []

    chosen = _pick_split(ds, [split, "validation", "train", "test"])
    if chosen is None:
        return []

    rows: List[Dict[str, str]] = []
    for i, ex in enumerate(chosen):
        if i >= max_samples:
            break
        article = ex.get("article") or ""
        highlights = ex.get("highlights") or ""
        if not article or not highlights:
            continue
        rows.append({"input_text": str(article), "output_text": str(highlights)})
    return rows


def load_rewrite(split: str = "train", max_samples: int = 200) -> List[Dict[str, str]]:
    """Returns paraphrase rows: sentence1 -> sentence2 (PAWS label==1)."""
    try:
        ds = load_dataset("paws", "labeled_final")
    except Exception as e:
        LOGGER.error("Failed to load paws: %s", e)
        return []

    chosen = _pick_split(ds, [split, "train", "validation", "test"])
    if chosen is None:
        return []

    rows: List[Dict[str, str]] = []
    count = 0
    for ex in chosen:
        if count >= max_samples:
            break
        if ex.get("label") != 1:
            continue
        s1 = ex.get("sentence1")
        s2 = ex.get("sentence2")
        if not s1 or not s2:
            continue
        rows.append({"input_text": str(s1), "output_text": str(s2)})
        count += 1
    return rows

