"""Shared inspect_ai defender model for AgentHarm (GPU 1 for local HF weights)."""

from __future__ import annotations

import torch
from inspect_ai.model import Model, get_model

from src.agentharm.message_bridge import to_inspect_model_name

_DEFENDER_MODEL: Model | None = None
_DEFENDER_CACHE_KEY: tuple[str, bool] | None = None


def is_openai_defender(target_model_name: str) -> bool:
    openai_prefixes = ("gpt-", "gpt-4", "o1", "o3", "o4")
    return any(target_model_name.startswith(prefix) for prefix in openai_prefixes)


def defender_device() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "cuda:1"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_defender_model(target_model_name: str, defender_temperature: float = 0.0) -> Model:
    global _DEFENDER_MODEL, _DEFENDER_CACHE_KEY

    inspect_name = to_inspect_model_name(target_model_name)
    do_sample = defender_temperature > 0.0
    cache_key = (inspect_name, do_sample)
    if _DEFENDER_MODEL is not None and _DEFENDER_CACHE_KEY == cache_key:
        return _DEFENDER_MODEL

    device = defender_device()
    if is_openai_defender(target_model_name):
        print(f"[AgentHarm] Loading OpenAI defender via inspect_ai: {inspect_name}")
        model = get_model(inspect_name)
    else:
        print(
            f"[AgentHarm] Loading HF defender via inspect_ai on {device}: {inspect_name} "
            f"(do_sample={do_sample})"
        )
        model = get_model(inspect_name, device=device, do_sample=do_sample)

    _DEFENDER_MODEL = model
    _DEFENDER_CACHE_KEY = cache_key
    return model
