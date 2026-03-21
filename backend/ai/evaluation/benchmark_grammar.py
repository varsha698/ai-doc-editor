from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark grammar correction accuracy.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/ai/evaluation/outputs"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    run_benchmark(
        task="grammar",
        split=args.split,
        limit=args.limit,
        output_dir=args.output_dir.resolve(),
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

