from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger("preprocess_data")


@dataclass
class DataConfig:
    prompts_dir: Path
    user_docs_dir: Path
    output_dir: Path
    max_chunk_chars: int = 1800
    overlap_chars: int = 200
    validation_ratio: float = 0.1
    seed: int = 42


def read_text_files(directory: Path) -> List[Path]:
    if not directory.exists():
        LOGGER.warning("Directory not found: %s", directory)
        return []
    return sorted([p for p in directory.rglob("*.txt") if p.is_file()])


def chunk_text(text: str, max_chunk_chars: int, overlap_chars: int) -> List[str]:
    if len(text) <= max_chunk_chars:
        return [text.strip()]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def infer_task_name(path: Path) -> str:
    stem = path.stem.lower()
    if "grammar" in stem:
        return "grammar_correction"
    if "rewrite" in stem:
        return "rewrite"
    if "outline" in stem:
        return "outline_generation"
    if "summary" in stem or "summar" in stem:
        return "summarization"
    return "style_adaptation"


def build_instruction_examples(config: DataConfig) -> List[dict]:
    examples: List[dict] = []
    prompt_files = read_text_files(config.prompts_dir)
    user_doc_files = read_text_files(config.user_docs_dir)

    for path in prompt_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        task = infer_task_name(path)
        examples.append(
            {
                "task": task,
                "instruction": f"Follow this prompt template for {task}.",
                "input": text,
                "output": "A high-quality response matching the task intent and preserving user meaning.",
                "metadata": {"source": str(path)},
            }
        )

    for path in user_doc_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = chunk_text(text, config.max_chunk_chars, config.overlap_chars)

        for para in paragraphs:
            chunks = chunk_text(para, config.max_chunk_chars, config.overlap_chars)
            for chunk in chunks:
                examples.extend(
                    [
                        {
                            "task": "grammar_correction",
                            "instruction": "Correct grammar while preserving original voice.",
                            "input": chunk,
                            "output": chunk,
                            "metadata": {"source": str(path)},
                        },
                        {
                            "task": "rewrite",
                            "instruction": "Rewrite this paragraph for clarity and flow.",
                            "input": chunk,
                            "output": chunk,
                            "metadata": {"source": str(path)},
                        },
                        {
                            "task": "summarization",
                            "instruction": "Summarize this section in 2-3 bullet points.",
                            "input": chunk,
                            "output": "- Key point 1\n- Key point 2",
                            "metadata": {"source": str(path)},
                        },
                    ]
                )
    return examples


def to_chatml_record(example: dict) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an editing assistant for professional document writing.",
            },
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
            {"role": "assistant", "content": example["output"]},
        ],
        "task": example["task"],
        "metadata": example.get("metadata", {}),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_val(records: List[dict], ratio: float, seed: int) -> tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess prompts and user docs for instruction tuning.")
    parser.add_argument("--prompts-dir", default="backend/ai/prompts")
    parser.add_argument("--user-docs-dir", default="backend/ai/datasets/user_docs")
    parser.add_argument("--output-dir", default="backend/ai/datasets/processed")
    parser.add_argument("--max-chunk-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    cfg = DataConfig(
        prompts_dir=Path(args.prompts_dir),
        user_docs_dir=Path(args.user_docs_dir),
        output_dir=Path(args.output_dir),
        max_chunk_chars=args.max_chunk_chars,
        overlap_chars=args.overlap_chars,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    examples = build_instruction_examples(cfg)
    if not examples:
        raise ValueError("No examples generated. Add files to prompts and user docs directories.")

    records = [to_chatml_record(e) for e in examples]
    train_rows, val_rows = split_train_val(records, cfg.validation_ratio, cfg.seed)
    write_jsonl(cfg.output_dir / "train.jsonl", train_rows)
    write_jsonl(cfg.output_dir / "val.jsonl", val_rows)

    LOGGER.info("Generated %d records (%d train / %d val)", len(records), len(train_rows), len(val_rows))
    LOGGER.info("Output directory: %s", cfg.output_dir)


if __name__ == "__main__":
    main()
