"""Factory for AgentHarmEnvironment."""

from typing import Any, Optional

from src.agentharm.env import AgentHarmEnvironment
from src.base import Environment


def build_agentharm_environment(
    batch_size: int,
    options: dict[str, Any],
    target_model_name: str,
    max_steps: int,
    max_new_tokens: Optional[int] = None,
    debug_mode: bool = False,
    evaluation_mode: bool = False,
) -> Environment:
    if not target_model_name:
        raise ValueError("agentharm environment requires environment.target_model")

    task_name = options.get("task_name", "harmful")
    if task_name not in ("harmful", "benign", "chat"):
        raise ValueError(f"Invalid agentharm task_name: {task_name}")

    if options.get("evaluation_mode", evaluation_mode):
        evaluation_mode = True
    split = options.get("split", "val")
    if evaluation_mode:
        split = options.get("eval_split", "test_public")

    opts = dict(options)
    opts["task_name"] = task_name
    opts["split"] = split

    temperature = float(options.get("defender_temperature", 0.0))
    tokens = max_new_tokens if max_new_tokens is not None else int(options.get("max_new_tokens", 4096))

    return AgentHarmEnvironment(
        batch_size=batch_size,
        max_steps=max_steps,
        target_model_name=target_model_name,
        options=opts,
        max_new_tokens=tokens,
        temperature=temperature,
        debug_mode=debug_mode,
        evaluation_mode=evaluation_mode,
    )
