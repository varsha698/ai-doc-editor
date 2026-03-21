from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def similarity_ratio(a: str, b: str) -> float:
    a_n = normalize_text(a).lower()
    b_n = normalize_text(b).lower()
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    return float(SequenceMatcher(None, a_n, b_n).ratio())


def _tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    # Simple whitespace tokenization; punctuation is kept as part of tokens.
    return text.split(" ")


def rouge_1_f1(pred: str, ref: str) -> float:
    pred_t = _tokenize(pred)
    ref_t = _tokenize(ref)
    if not pred_t or not ref_t:
        return 0.0
    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for t in pred_t:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in ref_t:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t, c in pred_counts.items():
        if t in ref_counts:
            overlap += min(c, ref_counts[t])
    precision = overlap / max(len(pred_t), 1)
    recall = overlap / max(len(ref_t), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _lcs_len(a: List[str], b: List[str]) -> int:
    # O(n*m) DP. Keep tokens reasonably small via truncation in callers.
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_t = _tokenize(pred)
    ref_t = _tokenize(ref)
    if not pred_t or not ref_t:
        return 0.0

    lcs = _lcs_len(pred_t, ref_t)
    prec = lcs / max(len(pred_t), 1)
    rec = lcs / max(len(ref_t), 1)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_grammar(pred: str, ref: str) -> Dict[str, float]:
    # Grammar task is typically a sentence-to-sentence transformation; use similarity ratio.
    return {
        "grammar_accuracy": similarity_ratio(pred, ref),
    }


def score_summarization(pred: str, ref: str) -> Dict[str, float]:
    # Use lightweight ROUGE variants + edit similarity.
    return {
        "rouge_1_f1": rouge_1_f1(pred, ref),
        "rouge_l_f1": rouge_l_f1(pred, ref),
        "summary_similarity": similarity_ratio(pred, ref),
    }


def score_rewrite(pred: str, ref: str) -> Dict[str, float]:
    return {
        "rewrite_rouge_l_f1": rouge_l_f1(pred, ref),
        "rewrite_similarity": similarity_ratio(pred, ref),
    }


def aggregate_summary(scores: List[Dict[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    keys = list(scores[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = [s.get(k, 0.0) for s in scores]
        out[k] = sum(vals) / max(len(vals), 1)
    return out

