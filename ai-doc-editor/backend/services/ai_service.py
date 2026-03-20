from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from services.suggestion_engine import build_scores, local_suggestions

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "ai" / "prompts"


@lru_cache(maxsize=1)
def _client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _llm_completion(system_prompt: str, user_content: str) -> str:
    client = _client()
    if client is None:
        return "LLM unavailable. Configure OPENAI_API_KEY for full AI responses."
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def analyze_document(text: str) -> Dict:
    prompt = _load_prompt("grammar_prompt.txt")
    llm_note = _llm_completion(prompt, text[:6000])
    suggestions = local_suggestions(text)
    scores = build_scores(text)
    summary = llm_note[:400] if llm_note else "Analysis complete."
    return {"suggestions": suggestions, "scores": scores, "summary": summary}


def chat_edit(message: str, text: str, command: str | None = None) -> Dict:
    prompt_name = "rewrite_prompt.txt"
    if command == "summarize":
        prompt_name = "outline_prompt.txt"
    system_prompt = _load_prompt(prompt_name)
    reply = _llm_completion(system_prompt, f"Instruction: {message}\n\nDocument:\n{text[:6000]}")
    suggestions = local_suggestions(text)
    return {"reply": reply or "I reviewed your text and suggest improving clarity and structure.", "suggestions": suggestions}
