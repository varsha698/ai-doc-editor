# Instruction-tuning pipeline

End-to-end LoRA fine-tuning for the document-editor local LLM. Schema:

| Field | Description |
|-------|-------------|
| `task` | `grammar`, `summarize`, `rewrite`, or custom |
| `input_text` | Model input / passage |
| `instruction` | What to do |
| `output_text` | Target completion |

## 1. Prepare data

From the **repository root** (paths default accordingly):

```bash
python backend/ai/training/prepare_dataset.py \
  --output-dir backend/ai/datasets/instruction \
  --sources grammar,summarize,rewrite,user_docs \
  --max-per-source 3000 \
  --max-user-docs-rows 2000
```

Outputs:

- `train.jsonl` / `val.jsonl`
- `dataset_manifest.json` (counts, sources, content hash)

Loaders:

- **grammar** ΓåÆ [jfleg](https://huggingface.co/datasets/jfleg)
- **summarize** ΓåÆ [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) (config `3.0.0`)
- **rewrite** ΓåÆ [paws](https://huggingface.co/datasets/paws) (`labeled_final`, paraphrase pairs with `label==1`)
- **user_docs** ΓåÆ `.txt` under `backend/ai/datasets/user_docs` (synthetic triples per chunk)

## 2. Train (GPU if available, CPU otherwise)

```bash
python backend/ai/training/train_llm.py \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --train-file backend/ai/datasets/instruction/train.jsonl \
  --val-file backend/ai/datasets/instruction/val.jsonl \
  --output-root backend/ai/models/checkpoints \
  --run-name editor-lora
```

Creates a **versioned** directory:

`backend/ai/models/checkpoints/v_<UTC-timestamp>_<run-name>/`

Contents:

- `checkpoint-*` (intermediate)
- `final_adapter/` ΓÇö adapter + tokenizer to point `LOCAL_LLM_ADAPTER_PATH` at
- `training_manifest.json` (started + completed metadata)

Use `--cpu` to force CPU training.

## 3. Evaluate

```bash
python backend/ai/training/evaluate_model.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter-path backend/ai/models/checkpoints/v_.../final_adapter \
  --val-file backend/ai/datasets/instruction/val.jsonl
```

Writes `eval_manifest.json` inside the adapter directory (eval loss and perplexity).
