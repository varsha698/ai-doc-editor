# AI Doc Editor

Production-oriented scaffold for an AI-powered document editor using:

- Frontend: Next.js + React + TypeScript + Tailwind + Tiptap
- Backend: FastAPI + Python
- AI: OpenAI-compatible LLM + sentence-transformers embeddings
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

## Environment

Set optional variables for full LLM responses:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Frontend optional env:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Local LLM Fine-Tuning (No OpenAI)

This project supports local open-source models (Mistral/LLaMA) via LoRA adapters.

### 1) Prepare dataset

Add style samples in:

`backend/ai/datasets/user_docs/*.txt`

Then generate instruction-tuning JSONL:

```bash
python backend/ai/preprocess_data.py \
  --prompts-dir backend/ai/prompts \
  --user-docs-dir backend/ai/datasets/user_docs \
  --output-dir backend/ai/datasets/processed
```

### 2) Fine-tune model

```bash
python backend/ai/train_llm.py \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --train-file backend/ai/datasets/processed/train.jsonl \
  --val-file backend/ai/datasets/processed/val.jsonl \
  --output-dir backend/ai/models/mistral-editor-lora
```

Training automatically uses GPU when available, otherwise CPU.

### 3) Integrate with backend

Set env variables:

```bash
LOCAL_LLM_BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
LOCAL_LLM_ADAPTER_PATH=backend/ai/models/mistral-editor-lora
LOCAL_LLM_MAX_NEW_TOKENS=256
LOCAL_LLM_TEMPERATURE=0.3
LOCAL_LLM_TOP_P=0.9
```

The backend `ai_service.py` now uses `backend/services/local_llm.py` for:

- grammar/style suggestions
- paragraph rewrites
- section summaries
- outline generation

No OpenAI API is required.
