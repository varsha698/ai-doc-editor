from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger("train_llm")


@dataclass
class TrainConfig:
    model_name: str
    train_file: Path
    val_file: Path
    output_dir: Path
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def render_chat_record(messages: List[Dict[str, str]]) -> str:
    # Generic text format, compatible across Mistral/LLaMA style instruct models.
    formatted = []
    for m in messages:
        formatted.append(f"<|{m['role']}|>\n{m['content']}")
    return "\n".join(formatted) + "\n<|assistant|>\n"


def build_hf_dataset(rows: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    texts = [render_chat_record(row["messages"]) for row in rows]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        tokens = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def select_device_flags() -> tuple[bool, bool]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return False, True
        return True, False
    return False, False


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for local open-source LLM")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--train-file", default="backend/ai/datasets/processed/train.jsonl")
    parser.add_argument("--val-file", default="backend/ai/datasets/processed/val.jsonl")
    parser.add_argument("--output-dir", default="backend/ai/models/mistral-editor-lora")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    fp16, bf16 = select_device_flags()
    cfg = TrainConfig(
        model_name=args.model_name,
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        output_dir=Path(args.output_dir),
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        fp16=fp16,
        bf16=bf16,
    )

    train_rows = read_jsonl(cfg.train_file)
    val_rows = read_jsonl(cfg.val_file)
    LOGGER.info("Loaded %d train and %d val records", len(train_rows), len(val_rows))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
    )
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

    train_ds = build_hf_dataset(train_rows, tokenizer, cfg.max_length)
    val_ds = build_hf_dataset(val_rows, tokenizer, cfg.max_length)

    args_train = TrainingArguments(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to=[],
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    LOGGER.info("Starting fine-tuning for model: %s", cfg.model_name)
    trainer.train()
    trainer.model.save_pretrained(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    LOGGER.info("Saved LoRA adapter and tokenizer to: %s", cfg.output_dir)


if __name__ == "__main__":
    main()
