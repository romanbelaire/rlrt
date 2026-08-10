"""Factory for AgentHarmGenEnvironment."""

from typing import Any, Optional

from src.agentharm_gen.env import AgentHarmGenEnvironment
from src.base import Environment


def build_agentharm_gen_environment(
    batch_size: int,
    options: dict[str, Any],
    target_model_name: str,
    max_steps: int,
    max_new_tokens: Optional[int] = None,
    debug_mode: bool = False,
    evaluation_mode: bool = False,
) -> Environment:
    if not target_model_name:
        raise ValueError("agentharm_gen environment requires environment.target_model")

    opts = dict(options)
    temperature = float(options.get("defender_temperature", 0.0))
    tokens = max_new_tokens if max_new_tokens is not None else int(options.get("max_new_tokens", 4096))

    return AgentHarmGenEnvironment(
        batch_size=batch_size,
        max_steps=max_steps,
        target_model_name=target_model_name,
        options=opts,
        max_new_tokens=tokens,
        temperature=temperature,
        debug_mode=debug_mode,
        evaluation_mode=evaluation_mode,
    )
