# AI Doc Editor

Production-oriented scaffold for an AI-powered document editor using:

- Frontend: Next.js + React + TypeScript + Tailwind + Tiptap
- Backend: FastAPI + Python
- AI: Local Hugging Face Transformers (LLaMA 3 / Mistral) + sentence-transformers embeddings
- Storage: PostgreSQL + vector DB-ready schema

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`.

## Run Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend runs on `http://localhost:8000`.

## Environment ΓÇö local LLM (Transformers)

Point the backend at a directory with Hugging FaceΓÇôformat weights (config.json, tokenizer, safetensors or bin shards). Switch families for string fallbacks when a tokenizer has no `chat_template`.

```bash
# Primary: path to local weights (recommended)
LOCAL_LLM_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct

# Or legacy alias / Hub id for development (downloads on first run)
# LOCAL_LLM_BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# llama3 | mistral | auto (auto: infer from config.json / path name)
LOCAL_LLM_MODEL_FAMILY=auto

# Optional LoRA (use the training run's final_adapter directory)
LOCAL_LLM_ADAPTER_PATH=backend/ai/models/checkpoints/v_<timestamp>_editor-lora/final_adapter

# Multi-model support (optional)
# Select the active model at runtime via env:
#   LOCAL_LLM_ACTIVE_MODEL=mistral|llama|mixtral
#
# You can also provide model-specific weights/adapters:
#   LOCAL_LLM_MISTRAL_MODEL_PATH, LOCAL_LLM_MISTRAL_ADAPTER_PATH
#   LOCAL_LLM_LLAMA_MODEL_PATH,   LOCAL_LLM_LLAMA_ADAPTER_PATH
#   LOCAL_LLM_MIXTRAL_MODEL_PATH, LOCAL_LLM_MIXTRAL_ADAPTER_PATH
#
# If the model-specific vars are not set, the backend falls back to
# LOCAL_LLM_MODEL_PATH / LOCAL_LLM_ADAPTER_PATH.

LOCAL_LLM_MAX_NEW_TOKENS=256

# Limit concurrent local generations (prevents GPU/CPU overload).
# Increase cautiously (typically 1 unless you know your hardware headroom).
LOCAL_LLM_MAX_CONCURRENT_GENERATIONS=1

LOCAL_LLM_TEMPERATURE=0.3
LOCAL_LLM_TOP_P=0.9
```

Optional: richer **offline grammar** checks (in addition to `textstat` + `pyspellchecker`):

```bash
pip install language-tool-python
```

On first use, LanguageTool may download its runtime; if it fails, the API still uses rules + spellcheck.

### RAG (ChromaDB)

Vector index is stored under `backend/chroma_data` by default (override with `CHROMA_PERSIST_DIR`).
Optional tuning: `VECTOR_CHUNK_CHARS`, `VECTOR_CHUNK_OVERLAP`, `VECTOR_TOP_K`.

For scalable background embedding indexing (request batching across documents):
- `VECTOR_INDEX_BATCH_WINDOW_MS` (default `250`)
- `VECTOR_INDEX_MAX_BATCH_TASKS` (default `12`)
- `VECTOR_INDEX_MAX_BATCH_CHUNKS` (default `600`)

### API

| Endpoint | Description |
|----------|-------------|
| `POST /ai/analyze` | JSON: suggestions, scores, summary (unchanged) |
| `POST /ai/chat` | JSON: `reply`, `suggestions` (unchanged) |
| `POST /ai/chat/stream` | SSE: `data: {"token":"..."}` then `{"done":true}` or `{"error":"..."}` |

Frontend optional env:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Local LLM fine-tuning (instruction + LoRA)

Pipeline lives in `backend/ai/training/`. See `backend/ai/training/README.md` for details.

### 1) Prepare instruction data

Schema per row: `input_text`, `instruction`, `output_text`, plus `task` and optional `metadata`.

```bash
python backend/ai/training/prepare_dataset.py \
  --output-dir backend/ai/datasets/instruction \
  --sources grammar,summarize,rewrite,user_docs \
  --max-per-source 3000
```

Add your own `.txt` samples under `backend/ai/datasets/user_docs/` when using `user_docs` in `--sources`.

### 2) Train (versioned checkpoints)

Each run creates `backend/ai/models/checkpoints/v_<UTC>_<run-name>/` with `checkpoint-*` steps and `final_adapter/`.

```bash
python backend/ai/training/train_llm.py \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --train-file backend/ai/datasets/instruction/train.jsonl \
  --val-file backend/ai/datasets/instruction/val.jsonl \
  --output-root backend/ai/models/checkpoints \
  --run-name editor-lora
```

GPU is used when available; pass `--cpu` to force CPU.

### 3) Evaluate

```bash
python backend/ai/training/evaluate_model.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter-path backend/ai/models/checkpoints/v_<timestamp>_editor-lora/final_adapter \
  --val-file backend/ai/datasets/instruction/val.jsonl
```

### 4) Run the API with the adapter

Set `LOCAL_LLM_MODEL_PATH` to the base weights and `LOCAL_LLM_ADAPTER_PATH` to that runΓÇÖs `final_adapter` folder.

The backend uses `backend/services/local_llm.py` for suggestions and chat. The legacy `backend/ai/preprocess_data.py` ChatML export is optional; new training expects the instruction JSONL from `prepare_dataset.py`.
