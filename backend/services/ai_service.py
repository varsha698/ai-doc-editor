from __future__ import annotations

from pathlib import Path
from typing import Dict

from services.local_llm import get_local_llm_service
from services.suggestion_engine import build_scores, local_suggestions

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "ai" / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


async def analyze_document(text: str) -> Dict:
    llm = get_local_llm_service()
    llm_summary = await llm.generate("summary", text[:6000])
    model_suggestions = await llm.suggest(text[:6000])
    suggestions = model_suggestions + local_suggestions(text)
    scores = build_scores(text)
    summary = llm_summary[:400] if llm_summary else "Analysis complete."
    return {"suggestions": suggestions, "scores": scores, "summary": summary}


async def chat_edit(message: str, text: str, command: str | None = None) -> Dict:
    llm = get_local_llm_service()
    task = "rewrite"
    if command == "summarize":
        task = "summary"
    elif command == "create-outline":
        task = "outline"
    elif command == "make-professional":
        task = "style"
    reply = await llm.generate(task, f"Instruction: {message}\n\nDocument:\n{text[:6000]}")
    suggestions = local_suggestions(text)
    return {"reply": reply or "I reviewed your text and suggest improving clarity and structure.", "suggestions": suggestions}
