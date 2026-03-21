from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from services.embedding_service import embed_texts

LOGGER = logging.getLogger(__name__)

STYLE_COLLECTION = "user_style_embeddings"
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CHROMA_DIR = _BACKEND_DIR / "chroma_data"
_persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(_DEFAULT_CHROMA_DIR))).expanduser().resolve()
_style_lock = threading.Lock()

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "at",
    "by",
    "with",
    "from",
    "that",
    "this",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "but",
    "if",
    "then",
    "than",
}


def _sanitize_key(raw: str) -> str:
    return re.sub(r"[^\w\-.:@]+", "_", (raw or "").strip())[:200] or "unknown_user"


@lru_cache(maxsize=1)
def _chroma_client() -> chromadb.PersistentClient:
    _persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_persist_dir))


def _get_style_collection():
    return _chroma_client().get_or_create_collection(name=STYLE_COLLECTION)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _tone(text: str) -> str:
    lower = text.lower()
    formal_hits = len(re.findall(r"\btherefore|moreover|however|furthermore|thus|hence\b", lower))
    casual_hits = len(re.findall(r"\bhey|hi|awesome|cool|really|kinda|gonna|wanna\b", lower))
    first_person = len(re.findall(r"\b(i|we|my|our)\b", lower))
    exclam = lower.count("!")
    if formal_hits >= casual_hits + 2:
        return "formal"
    if casual_hits + exclam >= formal_hits + 2:
        return "casual"
    if first_person > 8:
        return "conversational"
    return "neutral"


def _structure(sentences: List[str], text: str) -> str:
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    avg_para_sent = len(sentences) / max(len(paragraphs), 1)
    if avg_para_sent >= 6:
        return "long-form"
    if avg_para_sent <= 2.2:
        return "concise"
    return "balanced"


def generate_style_profile(documents: List[str]) -> Dict[str, Any]:
    corpus = "\n\n".join([d.strip() for d in documents if d and d.strip()])
    if not corpus:
        return {
            "avg_sentence_length": 14.0,
            "vocabulary_usage": {"top_terms": [], "type_token_ratio": 0.0},
            "tone": "neutral",
            "structure": "balanced",
        }

    sentences = _split_sentences(corpus)
    words = re.findall(r"\b[\w']+\b", corpus.lower())
    word_count = max(len(words), 1)
    avg_sent_len = word_count / max(len(sentences), 1)

    terms = [w for w in words if len(w) > 3 and w not in _STOPWORDS]
    counts = Counter(terms)
    top_terms = [w for w, _ in counts.most_common(16)]
    ttr = len(set(words)) / word_count

    return {
        "avg_sentence_length": round(avg_sent_len, 2),
        "vocabulary_usage": {
            "top_terms": top_terms,
            "type_token_ratio": round(ttr, 4),
        },
        "tone": _tone(corpus),
        "structure": _structure(sentences, corpus),
    }


def analyze_user_style(user_id: str, documents: List[str]) -> Dict[str, Any]:
    profile = generate_style_profile(documents)
    clean_user = _sanitize_key(user_id)
    texts = [d.strip()[:3000] for d in documents if d and d.strip()]
    if not texts:
        return profile
    try:
        vectors = embed_texts(texts)
        ids = [f"{clean_user}:style:{i}" for i in range(len(texts))]
        metadatas = [
            {
                "user_id": clean_user,
                "tone": profile["tone"],
                "structure": profile["structure"],
                "avg_sentence_length": str(profile["avg_sentence_length"]),
                "profile_json": json.dumps(profile, ensure_ascii=False),
                "chunk_index": i,
            }
            for i in range(len(texts))
        ]
        with _style_lock:
            col = _get_style_collection()
            col.delete(where={"user_id": clean_user})
            col.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    except Exception:
        LOGGER.exception("analyze_user_style: failed storing style embedding for %s", clean_user)
    return profile


def apply_style_constraints(prompt: str, style_profile: Dict[str, Any] | None) -> str:
    if not style_profile:
        return prompt
    vocab = style_profile.get("vocabulary_usage", {})
    top_terms = ", ".join(vocab.get("top_terms", [])[:8])
    constrained = (
        "User writing style profile:\n"
        f"- Tone: {style_profile.get('tone', 'neutral')}\n"
        f"- Structure: {style_profile.get('structure', 'balanced')}\n"
        f"- Average sentence length: {style_profile.get('avg_sentence_length', 14)} words\n"
        f"- Typical vocabulary: {top_terms if top_terms else 'n/a'}\n\n"
        "Style adaptation rules:\n"
        "- Preserve the user's voice and sentence rhythm.\n"
        "- Prefer vocabulary consistent with the profile.\n"
        "- Keep structural patterns (concise/balanced/long-form) unless instruction conflicts.\n\n"
        f"{prompt}"
    )
    return constrained
