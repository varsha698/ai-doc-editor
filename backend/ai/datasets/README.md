# Local LLM Dataset Structure

Use this structure to build an instruction-tuning dataset with style adaptation:

- `user_docs/` raw user writing samples (`.txt`)
- `processed/train.jsonl` generated training records
- `processed/val.jsonl` generated validation records

## Example

```text
backend/ai/datasets/
  user_docs/
    user_001_doc_01.txt
    user_001_doc_02.txt
  processed/
    train.jsonl
    val.jsonl
```

## JSONL record schema

Each line in `train.jsonl`/`val.jsonl`:

```json
{
  "messages": [
    {"role": "system", "content": "You are an editing assistant for professional document writing."},
    {"role": "user", "content": "Rewrite this paragraph for clarity...\n\n<INPUT_TEXT>"},
    {"role": "assistant", "content": "<TARGET_OUTPUT_TEXT>"}
  ],
  "task": "rewrite",
  "metadata": {"source": "backend/ai/datasets/user_docs/user_001_doc_01.txt"}
}
```
