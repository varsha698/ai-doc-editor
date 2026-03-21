from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from common import read_jsonl, record_to_messages, utc_now_iso, validate_record, write_manifest

LOGGER = logging.getLogger("ai.training.train")


@dataclass
class TrainConfig:
    model_name: str
    train_file: Path
    val_file: Path
    output_root: Path
    run_name: str
    max_length: int = 2048
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    fp16: bool = False
    bf16: bool = False
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    warmup_ratio: float = 0.03
    seed: int = 42


def make_versioned_run_dir(root: Path, run_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name.strip() or "lora")[:64]
    run_dir = root / f"v_{ts}_{safe}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def select_precision_flags() -> tuple[bool, bool]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return False, True
        return True, False
    return False, False


def tokenize_instruction_row(
    tokenizer: Any,
    row: Dict[str, Any],
    max_length: int,
) -> Dict[str, Any] | None:
    rec = validate_record(row)
    if not rec:
        return None
    messages = record_to_messages(rec)
    try:
        prompt_str = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_str = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception as e:
        LOGGER.warning("apply_chat_template failed: %s", e)
        return None

    full_enc = tokenizer(
        full_str,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    fids = full_enc["input_ids"]
    mask = full_enc["attention_mask"]

    prompt_enc = tokenizer(
        prompt_str,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    pids = prompt_enc["input_ids"]
    p_len = len(pids)
    if p_len > 0 and len(fids) >= p_len and fids[:p_len] == pids:
        labels = [-100] * p_len + fids[p_len:]
    else:
        LOGGER.debug("Label mask fallback (full sequence) for one example.")
        labels = list(fids)

    return {"input_ids": fids, "attention_mask": mask, "labels": labels}


def build_dataset(rows: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> Dataset:
    records: List[Dict[str, Any]] = []
    for row in rows:
        t = tokenize_instruction_row(tokenizer, row, max_length)
        if t:
            records.append(t)
    if not records:
        raise ValueError("No valid training rows after tokenization.")
    return Dataset.from_list(records)


class InstructionDataCollator:
    """Pad input_ids / attention_mask / labels (labels padded with -100)."""

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        batch_input: List[List[int]] = []
        batch_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for f in features:
            ids = list(f["input_ids"])
            ms = list(f["attention_mask"])
            lab = list(f["labels"])
            pad_len = max_len - len(ids)
            batch_input.append(ids + [pad_id] * pad_len)
            batch_mask.append(ms + [0] * pad_len)
            batch_labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA instruction tuning (Transformers + PEFT).")
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("backend/ai/datasets/instruction/train.jsonl"),
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("backend/ai/datasets/instruction/val.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("backend/ai/models/checkpoints"),
        help="Versioned run directories are created under this root.",
    )
    parser.add_argument("--run-name", type=str, default="lora-editor", help="Suffix for v_<timestamp>_<run_name>")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available (slow but portable).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    fp16, bf16 = select_precision_flags()
    if args.cpu:
        fp16, bf16 = False, False

    run_dir = make_versioned_run_dir(args.output_root.resolve(), args.run_name)
    LOGGER.info("Run directory: %s", run_dir)

    cfg = TrainConfig(
        model_name=args.model_name,
        train_file=args.train_file.resolve(),
        val_file=args.val_file.resolve(),
        output_root=args.output_root.resolve(),
        run_name=args.run_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fp16=fp16,
        bf16=bf16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
    )

    train_rows = read_jsonl(cfg.train_file)
    val_rows = read_jsonl(cfg.val_file)
    if not train_rows:
        raise SystemExit(f"No training rows in {cfg.train_file}")
    LOGGER.info("Loaded %d train rows; %d val rows", len(train_rows), len(val_rows))

    local_files = Path(cfg.model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, local_files_only=local_files)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float32
    if torch.cuda.is_available() and not args.cpu:
        torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    model_kw: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if torch.cuda.is_available() and not args.cpu:
        model_kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        local_files_only=local_files,
        **model_kw,
    )
    if not torch.cuda.is_available() or args.cpu:
        model = model.to(torch.device("cpu"))
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = build_dataset(train_rows, tokenizer, cfg.max_length)
    val_ds = build_dataset(val_rows, tokenizer, cfg.max_length) if val_rows else None

    collator = InstructionDataCollator(tokenizer)

    training_args_kw: Dict[str, Any] = dict(
        output_dir=str(run_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=5,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to=[],
        seed=cfg.seed,
        warmup_ratio=cfg.warmup_ratio,
    )
    if val_ds:
        training_args_kw["eval_strategy"] = "steps"
        training_args_kw["eval_steps"] = cfg.eval_steps
        training_args_kw["load_best_model_at_end"] = True
        training_args_kw["metric_for_best_model"] = "eval_loss"
        training_args_kw["greater_is_better"] = False
    else:
        training_args_kw["eval_strategy"] = "no"

    targs = TrainingArguments(**training_args_kw)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    write_manifest(
        run_dir / "training_manifest.json",
        {
            "phase": "started",
            "model_name": cfg.model_name,
            "train_file": str(cfg.train_file),
            "val_file": str(cfg.val_file),
            "run_name": cfg.run_name,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        },
    )

    LOGGER.info("Starting training (GPU=%s)", torch.cuda.is_available() and not args.cpu)
    trainer.train()

    final_dir = run_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    write_manifest(
        run_dir / "training_manifest.json",
        {
            "phase": "completed",
            "model_name": cfg.model_name,
            "train_file": str(cfg.train_file),
            "val_file": str(cfg.val_file),
            "run_name": cfg.run_name,
            "final_adapter": str(final_dir),
            "checkpoints_glob": "checkpoint-*",
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
            "train_samples": len(train_ds),
            "val_samples": len(val_ds) if val_ds else 0,
        },
    )
    LOGGER.info("Saved final adapter to %s", final_dir)


if __name__ == "__main__":
    main()
