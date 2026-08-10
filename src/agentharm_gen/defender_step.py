"""Run one defender turn with generated AgentHarm-Gen tools."""

import random
from typing import Any, Literal

from inspect_ai._eval.task.generate import task_generate
from inspect_ai._util.notgiven import NOT_GIVEN
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.solver import Generate, Solver, TaskState, chain, generate, solver

from inspect_evals.agentharm.agents.default_agent import (
    set_tool_choice,
    system_prompt_based_on_model,
)

from src.agentharm.message_bridge import EpisodeState
from src.agentharm_gen.harness.registry import load_all_tools


@solver
def setup_generated_tools_for_episode(n_irrelevant_tools: int = 0) -> Solver:
    """Inject target tools from episode metadata into the task state."""
    all_tools = load_all_tools()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        target_fns = state.metadata["target_functions"]
        tools_list = []
        for name in target_fns:
            tools_list.append(all_tools[name]())
        if n_irrelevant_tools > 0:
            irrelevant_names = [n for n in all_tools if n not in target_fns]
            k = min(n_irrelevant_tools, len(irrelevant_names))
            for name in random.sample(irrelevant_names, k):
                tools_list.append(all_tools[name]())
        random.shuffle(tools_list)
        state.tools = tools_list
        return state

    return solve


def build_generate_fn(
    defender_model: Model,
    max_tokens: int,
    temperature: float,
) -> Generate:
    if temperature <= 0.0:
        config = GenerateConfig(max_tokens=max_tokens, max_retries=3)
    else:
        config = GenerateConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=3,
        )

    async def generate_fn(
        state: TaskState,
        tool_calls: Literal["loop", "single", "none"] = "loop",
        **kwargs: Any,
    ) -> TaskState:
        merged = config.merge(kwargs)
        return await task_generate(
            model=defender_model,
            state=state,
            tool_calls=tool_calls,
            cache=NOT_GIVEN,
            config=merged,
        )

    return generate_fn


async def run_defender_turn(
    episode: EpisodeState,
    attacker_message: str,
    defender_model: Model,
    max_tokens: int,
    temperature: float,
    tool_choice: Literal["forced_first", "auto", "none"],
    system_prompt_template: str | None,
    prompt_technique: Literal["standard", "cot", "react", "refusal"],
    max_defender_tool_rounds: int,
    n_irrelevant_tools: int,
) -> EpisodeState:
    episode.append_user_message(attacker_message)
    state = episode.build_task_state(defender_model, message_limit=max_defender_tool_rounds)
    generate_fn = build_generate_fn(defender_model, max_tokens, temperature)

    setup_solver = setup_generated_tools_for_episode(n_irrelevant_tools)
    state = await setup_solver(state, generate_fn)

    solvers = []
    if not episode.system_prompt_applied:
        solvers.append(
            system_prompt_based_on_model(system_prompt_template, prompt_technique)
        )
        episode.system_prompt_applied = True

    if tool_choice == "forced_first":
        solvers.extend(
            [
                set_tool_choice("forced_first"),
                generate(tool_calls="single"),
                set_tool_choice("auto"),
                generate(tool_calls="loop"),
            ]
        )
    elif tool_choice == "none":
        solvers.extend([set_tool_choice("none"), generate(tool_calls="none")])
    else:
        solvers.extend([set_tool_choice(tool_choice), generate(tool_calls="loop")])

    plan = chain(*solvers)
    state = await plan(state, generate_fn)
    episode.task_state = state
    episode.sync_messages_from_task_state()
    return episode


def run_defender_turn_sync(
    episode: EpisodeState,
    attacker_message: str,
    defender_model: Model,
    max_tokens: int,
    temperature: float,
    tool_choice: Literal["forced_first", "auto", "none"],
    system_prompt_template: str | None,
    prompt_technique: Literal["standard", "cot", "react", "refusal"],
    max_defender_tool_rounds: int,
    n_irrelevant_tools: int,
) -> EpisodeState:
    import asyncio

    coro = run_defender_turn(
        episode,
        attacker_message,
        defender_model,
        max_tokens,
        temperature,
        tool_choice,
        system_prompt_template,
        prompt_technique,
        max_defender_tool_rounds,
        n_irrelevant_tools,
    )
    return asyncio.run(coro)
