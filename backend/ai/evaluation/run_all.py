from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all local LLM evaluation tasks.")
    parser.add_argument("--split-grammar", type=str, default="validation")
    parser.add_argument("--split-summarize", type=str, default="validation")
    parser.add_argument("--split-rewrite", type=str, default="train")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/ai/evaluation/outputs"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grammar = run_benchmark(
        task="grammar",
        split=args.split_grammar,
        limit=args.limit,
        output_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
    )
    summarize = run_benchmark(
        task="summarize",
        split=args.split_summarize,
        limit=args.limit,
        output_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
    )
    rewrite = run_benchmark(
        task="rewrite",
        split=args.split_rewrite,
        limit=args.limit,
        output_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
    )

    summary = {
        "grammar": grammar,
        "summarize": summarize,
        "rewrite": rewrite,
    }
    with (out_dir / "all_tasks_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

