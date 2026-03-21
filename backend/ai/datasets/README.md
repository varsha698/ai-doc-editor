# Local LLM datasets

## Instruction tuning (recommended)

Used by `backend/ai/training/prepare_dataset.py` ΓåÆ `backend/ai/training/train_llm.py`.

Layout:

```text
backend/ai/datasets/
  user_docs/
    user_001_doc_01.txt
  instruction/          # created by prepare_dataset --output-dir
    train.jsonl
    val.jsonl
    dataset_manifest.json
```

### JSONL record schema

Each line:

```json
{
  "task": "rewrite",
  "input_text": "Original passage...",
  "instruction": "Rewrite for clarity while preserving meaning.",
  "output_text": "Revised passage...",
  "metadata": {"source": "hf:paws", "idx": 0}
}
```

## Legacy ChatML export

The older `backend/ai/preprocess_data.py` still emits `messages`-style JSONL under `processed/`. New training expects the **instruction** schema above; re-run `prepare_dataset.py` or convert records before using `training/train_llm.py`.
