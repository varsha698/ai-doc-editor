from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterator, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

LOGGER = logging.getLogger("local_llm")


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configure the on-device causal LM.

    Switch models with ``LOCAL_LLM_MODEL_FAMILY`` (``llama3`` | ``mistral`` | ``auto``) and
    point ``LOCAL_LLM_MODEL_PATH`` at a directory containing Hugging Face-format weights.
    """

    model_path: str = os.getenv(
        "LOCAL_LLM_MODEL_PATH",
        os.getenv("LOCAL_LLM_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
    )
    model_family: str = os.getenv("LOCAL_LLM_MODEL_FAMILY", "auto").lower().strip()
    adapter_path: str = os.getenv("LOCAL_LLM_ADAPTER_PATH", "backend/ai/models/mistral-editor-lora")
    max_new_tokens: int = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.3"))
    top_p: float = float(os.getenv("LOCAL_LLM_TOP_P", "0.9"))
    chunk_chars: int = int(os.getenv("LOCAL_LLM_CHUNK_CHARS", "2000"))
    overlap_chars: int = int(os.getenv("LOCAL_LLM_OVERLAP_CHARS", "200"))


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


def _is_local_dir(ref: str) -> bool:
    p = Path(ref)
    return p.is_dir() and (p / "config.json").is_file()


def _read_config_architectures(model_path: str) -> List[str]:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return []
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        arch = data.get("architectures") or []
        return [str(a).lower() for a in arch]
    except (OSError, json.JSONDecodeError) as e:
        LOGGER.warning("Could not read model config at %s: %s", cfg_path, e)
        return []


def _effective_family(config: LocalLLMConfig, model_path: str) -> str:
    fam = config.model_family
    # Prompt family normalization.
    # - Mixtral uses the Mistral-style chat format in practice.
    # - LLaMA weights should use the LLaMA 3 chat markers when available.
    if fam in ("llama3", "llama"):
        return "llama3"
    if fam in ("mistral", "mixtral"):
        return "mistral"
    arch = _read_config_architectures(model_path)
    joined = " ".join(arch)
    if "mistral" in joined:
        return "mistral"
    if "llama" in joined:
        return "llama3"
    low = model_path.lower()
    if "mistral" in low:
        return "mistral"
    if "llama" in low or "meta-llama" in low:
        return "llama3"
    return "mistral"


def _fallback_chat_prompt(family: str, system: str, user: str) -> str:
    if family == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    # Mistral instruct-style default
    return f"<s>[INST] {system}\n\n{user} [/INST]"


class LocalLLMService:
    def __init__(self, config: LocalLLMConfig | None = None):
        self.config = config or LocalLLMConfig()
        self._family = _effective_family(self.config, self.config.model_path)
        LOGGER.info(
            "LocalLLMService init: path=%s family=%s (effective=%s)",
            self.config.model_path,
            self.config.model_family,
            self._family,
        )
        self.tokenizer, self.model = self._load_model()

    def _resolve_pretrained_source(self) -> str:
        ref = self.config.model_path.strip()
        if not ref:
            raise ValueError("LOCAL_LLM_MODEL_PATH / LOCAL_LLM_BASE_MODEL is empty.")
        if _is_local_dir(ref):
            LOGGER.info("Loading weights from local directory: %s", ref)
            return ref
        LOGGER.info("Loading weights from Hugging Face Hub id (dev fallback): %s", ref)
        return ref

    def _load_model(self) -> tuple[Any, Any]:
        src = self._resolve_pretrained_source()
        try:
            tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True, local_files_only=_is_local_dir(src))
        except Exception:
            LOGGER.exception("Failed to load tokenizer from %s", src)
            raise

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kw: Dict[str, Any] = {"local_files_only": _is_local_dir(src)}
        if torch.cuda.is_available():
            load_kw["torch_dtype"] = torch.float16
            load_kw["device_map"] = "auto"
        else:
            load_kw["torch_dtype"] = torch.float32

        try:
            LOGGER.info("Loading causal LM from %s", src)
            base_model = AutoModelForCausalLM.from_pretrained(src, **load_kw)
        except Exception:
            LOGGER.exception("Failed to load model weights from %s", src)
            raise

        adapter = self.config.adapter_path
        if adapter and os.path.isdir(adapter):
            LOGGER.info("Loading LoRA adapter from: %s", adapter)
            try:
                model = PeftModel.from_pretrained(base_model, adapter)
            except Exception:
                LOGGER.exception("Failed to load LoRA adapter; using base model only.")
                model = base_model
        else:
            if adapter:
                LOGGER.warning("Adapter path not found or not a directory: %s ΓÇö using base only.", adapter)
            model = base_model

        if not torch.cuda.is_available():
            model = model.to(torch.device("cpu"))

        model.eval()
        return tokenizer, model

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        max_len = getattr(self.tokenizer, "model_max_length", None) or 8192
        if max_len > 100_000:
            max_len = 8192
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        device = self._model_device()
        return {k: v.to(device) for k, v in inputs.items()}

    def _task_instruction(self, task: str) -> str:
        instructions = {
            "grammar": "Correct grammar and preserve writing style.",
            "rewrite": "Rewrite for clarity, flow, and stronger wording while preserving meaning.",
            "summary": "Summarize the section in concise bullet points.",
            "outline": "Generate a hierarchical outline for this content.",
            "style": "Adapt this to match the user's style profile.",
        }
        return instructions.get(task, "Improve this text.")

    def _messages_for_task(self, task: str, text: str) -> List[Dict[str, str]]:
        system = (
            "You are an AI writing assistant for a document editor. "
            "Follow the task precisely and answer with the improved or requested content only."
        )
        user = f"Task: {self._task_instruction(task)}\n\nInput:\n{text}\n\nProvide the output only."
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _format_prompt(self, task: str, text: str) -> str:
        messages = self._messages_for_task(task, text)
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if tmpl:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                LOGGER.warning("apply_chat_template failed (%s); using string fallback.", e)
        system = messages[0]["content"]
        user = messages[1]["content"]
        return _fallback_chat_prompt(self._family, system, user)

    def _generate_sync(self, prompt: str) -> str:
        inputs = self._prepare_inputs(prompt)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[-1]
        new_tokens = output[0, in_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _stream_tokens_sync(self, prompt: str) -> Iterator[str]:
        inputs = self._prepare_inputs(prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        err: list[BaseException | None] = [None]

        def run_generate() -> None:
            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except BaseException as e:
                LOGGER.exception("Streaming generation failed")
                err[0] = e

        thread = threading.Thread(target=run_generate, name="local-llm-generate", daemon=True)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join(timeout=600)
            if thread.is_alive():
                LOGGER.error("Generation thread did not finish within timeout.")
            if err[0] is not None:
                raise err[0]

    async def generate(self, task: str, text: str) -> str:
        chunks = chunk_text(text, self.config.chunk_chars, self.config.overlap_chars)
        if not chunks:
            return ""
        outputs: List[str] = []
        for chunk in chunks:
            prompt = self._format_prompt(task, chunk)
            try:
                chunk_out = await asyncio.to_thread(self._generate_sync, prompt)
            except Exception:
                LOGGER.exception("generate() failed for task=%s", task)
                raise
            outputs.append(chunk_out)
        return "\n".join(outputs).strip()

    async def stream_generate(self, task: str, text: str) -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        chunks = chunk_text(text, self.config.chunk_chars, self.config.overlap_chars)
        for chunk in chunks:
            prompt = self._format_prompt(task, chunk)
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

            def drain_stream(
                prompt_snapshot: str = prompt,
                q: asyncio.Queue[str | None] = queue,
            ) -> None:
                try:
                    for tok in self._stream_tokens_sync(prompt_snapshot):
                        asyncio.run_coroutine_threadsafe(q.put(tok), loop).result()
                finally:
                    asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

            worker = asyncio.create_task(asyncio.to_thread(drain_stream))
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                await worker

    async def suggest(self, text: str) -> List[Dict]:
        try:
            grammar_text = await self.generate("grammar", text)
            rewrite_text = await self.generate("rewrite", text)
            summary_text = await self.generate("summary", text)
        except Exception:
            LOGGER.exception("suggest() failed")
            raise

        return [
            {
                "id": str(uuid.uuid4()),
                "type": "grammar",
                "original_text": text[:250],
                "suggested_text": grammar_text[:600] or text[:250],
                "explanation": "Grammar and syntax improvements from local model.",
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


@lru_cache(maxsize=4)
def get_local_llm_service(config: LocalLLMConfig | None = None) -> LocalLLMService:
    """
    Cached factory for LocalLLMService.

    Cache key is the LocalLLMConfig instance, allowing multiple model families
    (Mistral/LLaMA/Mixtral) to coexist during a server's lifetime.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    cfg = config or LocalLLMConfig()
    return LocalLLMService(cfg)
