from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import AsyncGenerator, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

LOGGER = logging.getLogger("local_llm")


@dataclass(frozen=True)
class LocalLLMConfig:
    base_model: str = os.getenv("LOCAL_LLM_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    adapter_path: str = os.getenv("LOCAL_LLM_ADAPTER_PATH", "backend/ai/models/mistral-editor-lora")
    max_new_tokens: int = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.3"))
    top_p: float = float(os.getenv("LOCAL_LLM_TOP_P", "0.9"))
    chunk_chars: int = int(os.getenv("LOCAL_LLM_CHUNK_CHARS", "2000"))
    overlap_chars: int = int(os.getenv("LOCAL_LLM_OVERLAP_CHARS", "200"))


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


class LocalLLMService:
    def __init__(self, config: LocalLLMConfig | None = None):
        self.config = config or LocalLLMConfig()
        self.tokenizer, self.model = self._load_model()

    def _load_model(self):
        LOGGER.info("Loading base local model: %s", self.config.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if os.path.isdir(self.config.adapter_path):
            LOGGER.info("Loading LoRA adapter from: %s", self.config.adapter_path)
            model = PeftModel.from_pretrained(base_model, self.config.adapter_path)
        else:
            LOGGER.warning("Adapter not found at %s, using base model only.", self.config.adapter_path)
            model = base_model
        model.eval()
        return tokenizer, model

    def _build_prompt(self, task: str, text: str) -> str:
        instructions = {
            "grammar": "Correct grammar and preserve writing style.",
            "rewrite": "Rewrite for clarity, flow, and stronger wording while preserving meaning.",
            "summary": "Summarize the section in concise bullet points.",
            "outline": "Generate a hierarchical outline for this content.",
            "style": "Adapt this to match the user's style profile.",
        }
        instruction = instructions.get(task, "Improve this text.")
        return (
            "You are an AI writing assistant for a document editor.\n"
            f"Task: {instruction}\n\n"
            f"Input:\n{text}\n\n"
            "Output:"
        )

    def _generate_sync(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(_device())
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated.replace(prompt, "", 1).strip()

    async def generate(self, task: str, text: str) -> str:
        chunks = chunk_text(text, self.config.chunk_chars, self.config.overlap_chars)
        if not chunks:
            return ""
        outputs: List[str] = []
        for chunk in chunks:
            prompt = self._build_prompt(task, chunk)
            chunk_out = await asyncio.to_thread(self._generate_sync, prompt)
            outputs.append(chunk_out)
        return "\n".join(outputs).strip()

    async def stream_generate(self, task: str, text: str) -> AsyncGenerator[str, None]:
        chunks = chunk_text(text, self.config.chunk_chars, self.config.overlap_chars)
        for chunk in chunks:
            prompt = self._build_prompt(task, chunk)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(_device())
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs = {
                **inputs,
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": self.config.temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }

            thread = asyncio.to_thread(self.model.generate, **kwargs)
            task_runner = asyncio.create_task(thread)
            try:
                for token in streamer:
                    yield token
            finally:
                await task_runner

    async def suggest(self, text: str) -> List[Dict]:
        grammar_text = await self.generate("grammar", text)
        rewrite_text = await self.generate("rewrite", text)
        summary_text = await self.generate("summary", text)
        return [
            {
                "id": str(uuid.uuid4()),
                "type": "grammar",
                "original_text": text[:250],
                "suggested_text": grammar_text[:600] or text[:250],
                "explanation": "Grammar and syntax improvements from local fine-tuned model.",
            },
            {
                "id": str(uuid.uuid4()),
                "type": "clarity",
                "original_text": text[:250],
                "suggested_text": rewrite_text[:600] or text[:250],
                "explanation": "Rewritten for clarity and readability.",
            },
            {
                "id": str(uuid.uuid4()),
                "type": "structure",
                "original_text": text[:250],
                "suggested_text": summary_text[:600] or text[:250],
                "explanation": "Summarized and structured by local model.",
            },
        ]


@lru_cache(maxsize=1)
def get_local_llm_service() -> LocalLLMService:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    return LocalLLMService()
