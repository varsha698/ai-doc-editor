from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Set

from common import (
    dataset_content_hash,
    utc_now_iso,
    validate_record,
    write_jsonl,
    write_manifest,
)
from loaders import (
    load_grammar_correction_hf,
    load_rewrite_paraphrase_hf,
    load_summarization_hf,
    load_user_docs_local,
)

LOGGER = logging.getLogger("ai.training.prepare")


def _parse_sources(raw: str) -> Set[str]:
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    allowed = {"grammar", "summarize", "rewrite", "user_docs"}
    unknown = parts - allowed
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}. Use: {sorted(allowed)}")
    return parts


def gather_rows(
    sources: Set[str],
    max_per_source: int | None,
    user_docs_dir: Path | None,
    max_user_docs_rows: int | None,
    max_chunk_chars: int,
    overlap_chars: int,
    max_input_chars_grammar: int,
    max_input_chars_summary: int,
    max_input_chars_rewrite: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = max_per_source

    if "grammar" in sources:
        rows.extend(load_grammar_correction_hf(max_samples=n, max_input_chars=max_input_chars_grammar))
    if "summarize" in sources:
        rows.extend(load_summarization_hf(max_samples=n, max_input_chars=max_input_chars_summary))
    if "rewrite" in sources:
        rows.extend(load_rewrite_paraphrase_hf(max_samples=n, max_input_chars=max_input_chars_rewrite))
    if "user_docs" in sources and user_docs_dir:
        ud = load_user_docs_local(
            user_docs_dir,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
        if max_user_docs_rows is not None and len(ud) > max_user_docs_rows:
            ud = ud[:max_user_docs_rows]
        rows.extend(ud)

    cleaned: List[Dict[str, Any]] = []
    for r in rows:
        v = validate_record(r)
        if v:
            cleaned.append(dict(v))
    return cleaned


def split_train_val(
    records: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build instruction-tuning JSONL (input_text, instruction, output_text)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/ai/datasets/instruction"),
        help="Directory for train.jsonl, val.jsonl, dataset_manifest.json",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="grammar,summarize,rewrite",
        help="Comma-separated: grammar, summarize, rewrite, user_docs",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=5000,
        help="Max rows per HF source (ignored for user_docs unless you rely on size)",
    )
    parser.add_argument("--user-docs-dir", type=Path, default=Path("backend/ai/datasets/user_docs"))
    parser.add_argument("--no-user-docs", action="store_true", help="Omit user_docs even if in sources")
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chunk-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--max-input-chars-grammar", type=int, default=2000)
    parser.add_argument("--max-input-chars-summary", type=int, default=6000)
    parser.add_argument("--max-input-chars-rewrite", type=int, default=1500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    sources = _parse_sources(args.sources)
    if args.no_user_docs:
        sources.discard("user_docs")

    user_dir = args.user_docs_dir if "user_docs" in sources else None
    per_source = args.max_per_source if args.max_per_source > 0 else None
    max_ud = args.max_user_docs_rows if args.max_user_docs_rows > 0 else None

    rows = gather_rows(
        sources,
        per_source,
        user_dir,
        max_ud,
        args.max_chunk_chars,
        args.overlap_chars,
        args.max_input_chars_grammar,
        args.max_input_chars_summary,
        args.max_input_chars_rewrite,
    )
    if not rows:
        raise SystemExit(
            "No examples produced. Check network for HF datasets or add user_docs and include user_docs in --sources."
        )

    train_rows, val_rows = split_train_val(rows, args.validation_ratio, args.seed)
    out = args.output_dir.resolve()
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)

    manifest = {
        "format": {
            "fields": ["task", "input_text", "instruction", "output_text"],
            "optional": ["metadata"],
        },
        "created_utc": utc_now_iso(),
        "sources": sorted(sources),
        "counts": {"total": len(rows), "train": len(train_rows), "val": len(val_rows)},
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "max_per_source": args.max_per_source,
        "max_user_docs_rows": args.max_user_docs_rows,
        "content_hash_sample": dataset_content_hash(rows),
        "paths": {
            "train": str(out / "train.jsonl"),
            "val": str(out / "val.jsonl"),
        },
    }
    write_manifest(out / "dataset_manifest.json", manifest)
    LOGGER.info("Wrote %d train / %d val rows to %s", len(train_rows), len(val_rows), out)


if __name__ == "__main__":
    main()
