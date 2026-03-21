from __future__ import annotations

import logging
import os
import asyncio
from typing import Literal, Optional

from services.local_llm import LocalLLMConfig, LocalLLMService, get_local_llm_service

LOGGER = logging.getLogger(__name__)

ModelKey = Literal["mistral", "llama", "mixtral"]


def _normalize_key(raw: str | None) -> ModelKey | None:
    if not raw:
        return None
    r = raw.strip().lower()
    if r in ("mistral", "mistralai"):
        return "mistral"
    if r in ("llama", "llama3", "meta-llama"):
        return "llama"
    if r in ("mixtral", "mixtral8x7b"):
        return "mixtral"
    return None


def _get_model_key(request_override: str | None) -> ModelKey:
    normalized = _normalize_key(request_override)
    if normalized:
        return normalized

    env_key = _normalize_key(os.getenv("LOCAL_LLM_ACTIVE_MODEL"))
    if env_key:
        return env_key

    # Backward-compatible fallback: use LOCAL_LLM_MODEL_FAMILY if present.
    legacy_family = _normalize_key(os.getenv("LOCAL_LLM_MODEL_FAMILY"))
    if legacy_family:
        return legacy_family

    # Default to mistral.
    return "mistral"


def _pick_env(model: ModelKey, var_suffix: str) -> str | None:
    # Examples:
    # - model=mistral, var_suffix=MODEL_PATH => LOCAL_LLM_MISTRAL_MODEL_PATH
    # - model=llama, var_suffix=ADAPTER_PATH => LOCAL_LLM_LLAMA_ADAPTER_PATH
    prefix = f"LOCAL_LLM_{model.upper()}_{var_suffix}"
    return os.getenv(prefix)


def _build_config(model: ModelKey) -> LocalLLMConfig:
    # Start from the legacy/default config so temperature/top_p/chunk settings carry over.
    base = LocalLLMConfig()

    if model == "mistral":
        model_path = _pick_env(model, "MODEL_PATH") or base.model_path
        adapter_path = _pick_env(model, "ADAPTER_PATH") or base.adapter_path
        return LocalLLMConfig(
            model_path=model_path,
            model_family="mistral",
            adapter_path=adapter_path,
            max_new_tokens=base.max_new_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
            chunk_chars=base.chunk_chars,
            overlap_chars=base.overlap_chars,
        )

    if model == "llama":
        model_path = _pick_env(model, "MODEL_PATH") or base.model_path
        adapter_path = _pick_env(model, "ADAPTER_PATH") or base.adapter_path
        return LocalLLMConfig(
            model_path=model_path,
            model_family="llama3",
            adapter_path=adapter_path,
            max_new_tokens=base.max_new_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
            chunk_chars=base.chunk_chars,
            overlap_chars=base.overlap_chars,
        )

    # mixtral
    model_path = _pick_env(model, "MODEL_PATH") or base.model_path
    adapter_path = _pick_env(model, "ADAPTER_PATH") or base.adapter_path
    return LocalLLMConfig(
        model_path=model_path,
        model_family="mixtral",
        adapter_path=adapter_path,
        max_new_tokens=base.max_new_tokens,
        temperature=base.temperature,
        top_p=base.top_p,
        chunk_chars=base.chunk_chars,
        overlap_chars=base.overlap_chars,
    )


def get_active_local_llm(request_override: str | None = None) -> LocalLLMService:
    model_key = _get_model_key(request_override)
    cfg = _build_config(model_key)
    LOGGER.info("Using local LLM model=%s path=%s", model_key, cfg.model_path)
    return get_local_llm_service(cfg)


_generation_semaphores: dict[ModelKey, asyncio.Semaphore] = {}


def get_active_model_key(request_override: str | None = None) -> ModelKey:
    return _get_model_key(request_override)


def get_generation_semaphore(model_key: ModelKey) -> asyncio.Semaphore:
    limit = int(os.getenv("LOCAL_LLM_MAX_CONCURRENT_GENERATIONS", "1"))
    if limit < 1:
        limit = 1
    sem = _generation_semaphores.get(model_key)
    if sem is None:
        sem = asyncio.Semaphore(limit)
        _generation_semaphores[model_key] = sem
    return sem

