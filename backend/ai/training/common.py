from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, TypedDict

LOGGER = logging.getLogger("ai.training")

SCHEMA_VERSION = "1.0"
SYSTEM_PROMPT = (
    "You are an AI writing assistant for a document editor. "
    "Follow the instruction precisely and respond with the requested output only."
)


class InstructionRecord(TypedDict):
    task: str
    input_text: str
    instruction: str
    output_text: str
    metadata: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_record(row: Dict[str, Any]) -> InstructionRecord | None:
    for key in ("input_text", "instruction", "output_text"):
        if key not in row or not str(row.get(key, "")).strip():
            LOGGER.warning("Skipping invalid row (missing %s): %s", key, row.get("metadata"))
            return None
    task = str(row.get("task", "general")).strip()
    return {
        "task": task,
        "input_text": str(row["input_text"]).strip(),
        "instruction": str(row["instruction"]).strip(),
        "output_text": str(row["output_text"]).strip(),
        "metadata": dict(row.get("metadata") or {}),
    }


def record_to_messages(rec: InstructionRecord) -> List[Dict[str, str]]:
    user = f"{rec['instruction']}\n\nInput:\n{rec['input_text']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": rec["output_text"]},
    ]


def dataset_content_hash(rows: List[Dict[str, Any]], sample: int = 2000) -> str:
    h = hashlib.sha256()
    for row in rows[:sample]:
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    h.update(str(len(rows)).encode("utf-8"))
    return h.hexdigest()[:16]


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {**payload, "schema_version": SCHEMA_VERSION, "updated_utc": utc_now_iso()}
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
