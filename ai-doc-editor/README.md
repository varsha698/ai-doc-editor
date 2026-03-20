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
