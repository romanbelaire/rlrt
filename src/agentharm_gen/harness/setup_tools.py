"""Solver that injects generated @tool instances from the dynamic registry.

Mirrors inspect_evals' setup_tools_from_metadata but sources from
generated/tools/ instead of the static harmful_tools/ module.
"""

import random

from inspect_ai.solver import Generate, Solver, TaskState, solver

from src.agentharm_gen.harness.registry import load_all_tools


@solver
def setup_generated_tools(n_irrelevant_tools: int = 0) -> Solver:
    """Inject target tools from state.metadata['target_functions'] into the task state.

    Loads the complete generated tool registry once at call time, then for each
    sample selects the target tools + optional random irrelevant tools.

    Args:
        n_irrelevant_tools: Number of irrelevant (non-target) tools to add as distractors.
    """
    all_tools = load_all_tools()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        target_fns = state.metadata["target_functions"]
        tools_list = []

        for name in target_fns:
            if name not in all_tools:
                raise ValueError(
                    f"Target function '{name}' not found in generated tool registry. "
                    "Run: python -m src.agentharm_gen.cli convert"
                )
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
