from __future__ import annotations

import logging
import re
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Literal, TypedDict

LOGGER = logging.getLogger(__name__)

IssueType = Literal["grammar", "punctuation", "spelling", "clarity"]


class GrammarIssue(TypedDict, total=False):
    type: IssueType
    start: int
    end: int
    suggestion: str
    message: str


_lt_singleton: Any = None
_lt_failed: bool = False


def _get_language_tool() -> Any | None:
    global _lt_singleton, _lt_failed
    if _lt_failed:
        return None
    if _lt_singleton is not None:
        return _lt_singleton
    try:
        from language_tool_python import LanguageTool

        _lt_singleton = LanguageTool("en-US")
        LOGGER.info("LanguageTool (en-US) loaded for grammar checks.")
        return _lt_singleton
    except Exception as e:
        _lt_failed = True
        LOGGER.info("LanguageTool not available (%s); using rules + spellcheck only.", e)
        return None


@lru_cache(maxsize=1)
def _spell_checker() -> Any:
    from spellchecker import SpellChecker

    return SpellChecker(distance=1)


def _flesch_readability(text: str) -> float:
    try:
        import textstat

        score = float(textstat.flesch_reading_ease(text))
    except Exception as e:
        LOGGER.debug("textstat readability failed: %s", e)
        return 50.0
    if score > 100:
        return 100.0
    if score < 0:
        return 0.0
    return score


def _clarity_score(text: str) -> float:
    """Heuristic 0ΓÇô100: sentence length, long words, passive-ish patterns."""
    words = re.findall(r"\b[\w']+\b", text)
    if not words:
        return 70.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    n_sent = max(len(sentences), 1)
    avg_sent_words = len(words) / n_sent
    long_words = sum(1 for w in words if len(w) > 6)
    long_ratio = long_words / len(words)
    passive_hits = len(re.findall(r"\b(?:is|are|was|were)\s+\w+ed\b", text, re.IGNORECASE))
    passive_ratio = passive_hits / n_sent

    penalty = 0.0
    penalty += min(35.0, abs(avg_sent_words - 17) * 1.2)
    penalty += min(25.0, long_ratio * 80)
    penalty += min(20.0, passive_ratio * 25)
    return max(0.0, min(100.0, 100.0 - penalty))


def _grammar_score_from_counts(grammar: int, spell: int, punct: int, clarity_issues: int) -> int:
    raw = 100.0 - (3.5 * grammar + 2.5 * spell + 1.5 * punct + 2.0 * clarity_issues)
    return int(max(15, min(100, round(raw))))


def _lt_category_to_type(category: str) -> IssueType:
    c = (category or "").upper()
    if "PUNCTUATION" in c or "TYPOGRAPHY" in c:
        return "punctuation"
    if "STYLE" in c or "REDUNDANCY" in c:
        return "clarity"
    return "grammar"


def _language_tool_issues(text: str) -> List[GrammarIssue]:
    tool = _get_language_tool()
    if not tool:
        return []
    try:
        matches = tool.check(text)
    except Exception as e:
        LOGGER.warning("LanguageTool check failed: %s", e)
        return []

    out: List[GrammarIssue] = []
    for m in matches:
        start = int(getattr(m, "offset", 0) or 0)
        length = int(getattr(m, "errorLength", 0) or 0)
        end = min(start + length, len(text))
        if start < 0 or end <= start:
            continue
        reps = getattr(m, "replacements", None) or []
        suggestion = str(reps[0]) if reps else text[start:end]
        msg = str(getattr(m, "message", "") or "").strip()
        rule_id = str(getattr(m, "ruleId", "") or "")
        cat = str(getattr(m, "category", "") or "")
        issue_type = _lt_category_to_type(cat or rule_id)
        if "SPELL" in rule_id.upper() or "MORFOLOGIK" in rule_id.upper():
            issue_type = "spelling"
        out.append(
            {
                "type": issue_type,
                "start": start,
                "end": end,
                "suggestion": suggestion,
                "message": msg or (f"Grammar ({rule_id})" if rule_id else "Grammar"),
            }
        )
    return out


def _spelling_issues(text: str, max_issues: int = 25) -> List[GrammarIssue]:
    spell = _spell_checker()
    issues: List[GrammarIssue] = []
    seen: set[tuple[int, int]] = set()
    for m in re.finditer(r"\b[\w']+\b", text):
        raw = m.group()
        if len(raw) <= 1 or raw.isdigit():
            continue
        if raw.isupper() and len(raw) > 1:
            continue
        key = raw.lower()
        if not spell.unknown([key]):
            continue
        corr = spell.correction(key)
        if not corr or corr == key:
            continue
        start, end = m.start(), m.end()
        if (start, end) in seen:
            continue
        seen.add((start, end))
        original = text[start:end]
        suggestion = corr[:1] + corr[1:] if original[:1].isupper() else corr
        issues.append(
            {
                "type": "spelling",
                "start": start,
                "end": end,
                "suggestion": suggestion,
                "message": f'Possible misspelling of "{original}"',
            }
        )
        if len(issues) >= max_issues:
            break
    return issues


def _punctuation_issues(text: str) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    for m in re.finditer(r"  +", text):
        issues.append(
            {
                "type": "punctuation",
                "start": m.start(),
                "end": m.end(),
                "suggestion": " ",
                "message": "Multiple spaces; use a single space.",
            }
        )
    for m in re.finditer(r"\s+([,.;:!?])", text):
        issues.append(
            {
                "type": "punctuation",
                "start": m.start(),
                "end": m.end(),
                "suggestion": m.group(1),
                "message": "Remove space before punctuation.",
            }
        )
    for m in re.finditer(r"(?<=[a-zA-Z])([,;])(?=[A-Za-z])", text):
        issues.append(
            {
                "type": "punctuation",
                "start": m.start(1),
                "end": m.end(1),
                "suggestion": m.group(1) + " ",
                "message": "Add a space after comma or semicolon.",
            }
        )
    for m in re.finditer(r"\.{4,}", text):
        issues.append(
            {
                "type": "punctuation",
                "start": m.start(),
                "end": m.end(),
                "suggestion": "...",
                "message": "Use at most an ellipsis (...).",
            }
        )
    return issues


_CONTRACTIONS = [
    (r"\bdont\b", "don't"),
    (r"\bwont\b", "won't"),
    (r"\bcant\b", "can't"),
    (r"\bisnt\b", "isn't"),
    (r"\barent\b", "aren't"),
    (r"\bwasnt\b", "wasn't"),
    (r"\bwerent\b", "weren't"),
    (r"\bdoesnt\b", "doesn't"),
    (r"\bdidnt\b", "didn't"),
    (r"\bhasnt\b", "hasn't"),
    (r"\bhavent\b", "haven't"),
    (r"\bhadnt\b", "hadn't"),
    (r"\bwouldnt\b", "wouldn't"),
    (r"\bcouldnt\b", "couldn't"),
    (r"\bshouldnt\b", "shouldn't"),
    (r"\btheyre\b", "they're"),
    (r"\byoure\b", "you're"),
]


def _contraction_issues(text: str) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    for pattern, fix in _CONTRACTIONS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            issues.append(
                {
                    "type": "grammar",
                    "start": m.start(),
                    "end": m.end(),
                    "suggestion": fix,
                    "message": "Use an apostrophe in this contraction.",
                }
            )
    return issues


def _clarity_span_issues(text: str, max_issues: int = 12) -> List[GrammarIssue]:
    """Flag very long lines (often run-on sentences) as clarity hints."""
    issues: List[GrammarIssue] = []
    for m in re.finditer(r"[^\n]+", text):
        seg = m.group().strip()
        if not seg:
            continue
        words = len(re.findall(r"\b[\w']+\b", seg))
        if words >= 45:
            issues.append(
                {
                    "type": "clarity",
                    "start": m.start(),
                    "end": m.end(),
                    "suggestion": seg,
                    "message": f"Long line (~{words} words); consider shorter sentences.",
                }
            )
        if len(issues) >= max_issues:
            break
    return issues


def _dedupe_issues(issues: List[GrammarIssue]) -> List[GrammarIssue]:
    """Drop overlaps, prefer earlier higher-priority types."""
    priority = {"grammar": 0, "spelling": 1, "punctuation": 2, "clarity": 3}
    sorted_issues = sorted(issues, key=lambda x: (x["start"], priority.get(x["type"], 9)))
    kept: List[GrammarIssue] = []
    for cur in sorted_issues:
        a, b = cur["start"], cur["end"]
        overlap = False
        for prev in kept:
            pa, pb = prev["start"], prev["end"]
            if not (b <= pa or a >= pb):
                overlap = True
                break
        if not overlap:
            kept.append(cur)
    return sorted(kept, key=lambda x: x["start"])


def _suggestion_type(t: IssueType) -> str:
    if t == "clarity":
        return "clarity"
    return "grammar"


def issues_to_suggestions(text: str, issues: List[GrammarIssue]) -> List[Dict[str, Any]]:
    """Map engine issues to the editor suggestion card shape."""
    out: List[Dict[str, Any]] = []
    for issue in issues:
        start, end = issue["start"], issue["end"]
        span = text[start:end] if 0 <= start <= end <= len(text) else ""
        out.append(
            {
                "id": str(uuid.uuid4()),
                "type": _suggestion_type(issue["type"]),
                "original_text": span,
                "suggested_text": issue.get("suggestion", ""),
                "explanation": issue.get("message", ""),
                "start": start,
                "end": end,
            }
        )
    return out


def analyze_grammar(text: str) -> Dict[str, Any]:
    """
    Fast grammar/punctuation pass (LanguageTool optional + rules + pyspellchecker).
    Returns issues with character offsets, readability/clarity scores, and grammar score.
    """
    if not text or not text.strip():
        return {
            "issues": [],
            "readability_score": 70,
            "clarity_score": 70,
            "grammar_score": 90,
        }

    issues: List[GrammarIssue] = []
    issues.extend(_language_tool_issues(text))

    punct = _punctuation_issues(text)
    issues.extend(punct)

    contr = _contraction_issues(text)
    issues.extend(contr)

    spell = _spelling_issues(text)
    issues.extend(spell)

    clarity_spans = _clarity_span_issues(text)
    issues.extend(clarity_spans)

    issues = _dedupe_issues(issues)

    n_grammar = sum(1 for i in issues if i["type"] == "grammar")
    n_spell = sum(1 for i in issues if i["type"] == "spelling")
    n_punct = sum(1 for i in issues if i["type"] == "punctuation")
    n_clarity_i = sum(1 for i in issues if i["type"] == "clarity")

    grammar_score = _grammar_score_from_counts(n_grammar, n_spell, n_punct, n_clarity_i)

    readability = int(round(_flesch_readability(text)))
    clarity = int(round(_clarity_score(text)))

    return {
        "issues": [dict(i) for i in issues],
        "readability_score": readability,
        "clarity_score": clarity,
        "grammar_score": grammar_score,
    }
