from __future__ import annotations

import argparse
import logging
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from common import read_jsonl, utc_now_iso, write_manifest
from train_llm import InstructionDataCollator, build_dataset, select_precision_flags

LOGGER = logging.getLogger("ai.training.evaluate")


def _sample_rows(rows: List[Dict[str, Any]], max_samples: int | None) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return rows
    return rows[:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on instruction JSONL (masked CE loss).")
    parser.add_argument("--base-model", type=str, required=True, help="Base model id or local path.")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="PEFT adapter directory (e.g. .../checkpoints/v_.../final_adapter).",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("backend/ai/datasets/instruction/val.jsonl"),
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use full val set.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    val_path = args.val_file.resolve()
    adapter_path = args.adapter_path.resolve()
    if not adapter_path.is_dir():
        raise SystemExit(f"Adapter path is not a directory: {adapter_path}")

    rows = read_jsonl(val_path)
    if not rows:
        raise SystemExit(f"No rows in {val_path}")
    max_s = args.max_samples if args.max_samples > 0 else None
    rows = _sample_rows(rows, max_s)
    LOGGER.info("Evaluating on %d rows from %s", len(rows), val_path)

    fp16, bf16 = select_precision_flags()
    if args.cpu:
        fp16, bf16 = False, False

    local_base = Path(args.base_model).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=local_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float32
    if torch.cuda.is_available() and not args.cpu:
        torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    model_kw: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if torch.cuda.is_available() and not args.cpu:
        model_kw["device_map"] = "auto"
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=local_base,
        **model_kw,
    )
    if not torch.cuda.is_available() or args.cpu:
        base = base.to(torch.device("cpu"))

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    eval_ds = build_dataset(rows, tokenizer, args.max_length)
    collator = InstructionDataCollator(tokenizer)

    with tempfile.TemporaryDirectory() as tmp:
        targs = TrainingArguments(
            output_dir=tmp,
            per_device_eval_batch_size=args.batch_size,
            fp16=fp16,
            bf16=bf16,
            report_to=[],
        )
        trainer = Trainer(model=model, args=targs, eval_dataset=eval_ds, data_collator=collator)
        metrics = trainer.evaluate()

    loss = metrics.get("eval_loss")
    perplexity = math.exp(loss) if loss is not None and math.isfinite(loss) else None
    LOGGER.info("eval_loss=%s perplexity=%s", loss, perplexity)

    out = {
        "evaluated_utc": utc_now_iso(),
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "val_file": str(val_path),
        "num_samples": len(eval_ds),
        "max_length": args.max_length,
        "metrics": metrics,
        "perplexity": perplexity,
    }
    write_manifest(adapter_path / "eval_manifest.json", out)
    LOGGER.info("Wrote %s", adapter_path / "eval_manifest.json")


if __name__ == "__main__":
    main()
