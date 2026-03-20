from __future__ import annotations

import re
import uuid
from typing import Dict, List


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def build_scores(text: str) -> Dict[str, int]:
    word_count = max(len(text.split()), 1)
    sentence_count = max(len(_split_sentences(text)), 1)
    avg_sentence_len = word_count / sentence_count
    readability = max(10, min(100, int(100 - abs(avg_sentence_len - 18) * 3)))
    grammar = 90 if "  " not in text else 75
    clarity = max(20, min(100, int(100 - abs(avg_sentence_len - 14) * 4)))
    argument_strength = 70 if sentence_count > 3 else 55
    return {
        "readability": readability,
        "grammar": grammar,
        "clarity": clarity,
        "argument_strength": argument_strength,
    }


def local_suggestions(text: str) -> List[Dict]:
    suggestions: List[Dict] = []
    patterns = [
        (r"\butilize\b", "use", "Prefer simpler wording for clarity.", "style"),
        (r"\bin order to\b", "to", "Shorten phrasing to reduce verbosity.", "clarity"),
        (r"\bvery\b", "", "Avoid weak intensifiers when possible.", "style"),
    ]
    for pattern, replacement, explanation, suggestion_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            original = match.group(0)
            suggested = replacement if replacement else "(remove word)"
            suggestions.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": suggestion_type,
                    "original_text": original,
                    "suggested_text": suggested,
                    "explanation": explanation,
                }
            )
    return suggestions[:5]
