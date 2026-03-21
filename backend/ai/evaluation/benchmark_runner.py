from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Literal, Tuple

from services.local_llm import get_local_llm_service

from .datasets import (
    load_grammar_correction,
    load_rewrite,
    load_summarization,
)
from .scoring import (
    aggregate_summary,
    score_grammar,
    score_rewrite,
    score_summarization,
)

LOGGER = logging.getLogger(__name__)


TaskName = Literal["grammar", "summarize", "rewrite"]


@dataclass
class ExampleResult:
    input_text: str
    reference: str
    prediction: str
    latency_ms: float
    scores: Dict[str, float]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _percentiles(values: List[float], ps: List[int]) -> Dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    sv = sorted(values)
    out: Dict[str, float] = {}
    n = len(sv)
    for p in ps:
        k = int(round((p / 100.0) * (n - 1)))
        out[f"p{p}"] = float(sv[max(0, min(n - 1, k))])
    return out


def _truncate_for_metrics(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3].rstrip() + "..."


def _score_row(task: TaskName, pred: str, ref: str) -> Dict[str, float]:
    pred_t = _truncate_for_metrics(pred, 2000)
    ref_t = _truncate_for_metrics(ref, 2000)
    if task == "grammar":
        return score_grammar(pred_t, ref_t)
    if task == "summarize":
        return score_summarization(pred_t, ref_t)
    if task == "rewrite":
        return score_rewrite(pred_t, ref_t)
    raise ValueError(f"Unknown task {task}")


def _load_task_rows(task: TaskName, split: str, limit: int) -> List[Dict[str, str]]:
    if task == "grammar":
        return load_grammar_correction(split=split, max_samples=limit)
    if task == "summarize":
        return load_summarization(split=split, max_samples=limit)
    if task == "rewrite":
        return load_rewrite(split=split, max_samples=limit)
    raise ValueError(f"Unknown task {task}")


def _llm_task_name(task: TaskName) -> str:
    if task == "grammar":
        return "grammar"
    if task == "summarize":
        return "summary"
    if task == "rewrite":
        return "rewrite"
    raise ValueError(f"Unknown task {task}")


def run_benchmark(
    *,
    task: TaskName,
    split: str,
    limit: int,
    output_dir: Path,
    max_new_tokens: int,
) -> Dict[str, Any]:
    _ensure_dir(output_dir)

    # Let LocalLLMConfig read these env vars on first init in this process.
    os.environ["LOCAL_LLM_MAX_NEW_TOKENS"] = str(max_new_tokens)

    rows = _load_task_rows(task, split, limit)
    if not rows:
        raise SystemExit(f"No rows loaded for task={task} split={split}")

    llm = get_local_llm_service()

    results: List[ExampleResult] = []
    latencies: List[float] = []

    import asyncio

    async def _predict_one(input_text: str) -> str:
        return await llm.generate(_llm_task_name(task), input_text)

    t0 = time.time()
    for row in rows:
        inp = row["input_text"]
        ref = row["output_text"]

        t_start = time.time()
        pred = asyncio.run(_predict_one(inp))
        latency_ms = (time.time() - t_start) * 1000.0

        scores = _score_row(task, pred, ref)
        results.append(
            ExampleResult(
                input_text=inp,
                reference=ref,
                prediction=pred,
                latency_ms=latency_ms,
                scores=scores,
            )
        )
        latencies.append(latency_ms)

    total_s = time.time() - t0
    avg_latency = mean(latencies) if latencies else 0.0

    agg = aggregate_summary([r.scores for r in results])
    p = _percentiles(latencies, [50, 90, 95])

    payload = {
        "task": task,
        "split": split,
        "limit": limit,
        "avg_latency_ms": avg_latency,
        "latency_percentiles_ms": p,
        "total_seconds": total_s,
        "num_examples": len(results),
        "aggregated_scores": agg,
        "examples_path": str(output_dir / f"{task}_results.jsonl"),
    }

    with (output_dir / f"{task}_results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({**asdict(r), "scores": r.scores}, ensure_ascii=False) + "\n")

    with (output_dir / f"{task}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    LOGGER.info("Completed benchmark task=%s. Summary written to %s", task, output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local LLM evaluation benchmark.")
    parser.add_argument("--task", choices=["grammar", "summarize", "rewrite"], required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/ai/evaluation/outputs"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_benchmark(
        task=args.task,  # type: ignore[arg-type]
        split=args.split,
        limit=args.limit,
        output_dir=args.output_dir.resolve(),
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

