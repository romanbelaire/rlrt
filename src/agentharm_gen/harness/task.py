"""Inspect-AI task definition for the generated AgentHarm-Gen benchmark.

Run:
    inspect eval src/agentharm_gen/harness/task.py@agentharm_gen \\
        --model openai/gpt-4o --limit 5 \\
        -T semantic_judge=openai/gpt-4o-2024-08-06

Or via CLI:
    python -m src.agentharm_gen.cli eval --model openai/gpt-4o
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, get_model

from inspect_evals.agentharm.agents.agent import get_agent
from inspect_evals.agentharm.scorer import RefusalJudgeLLM, combined_scorer

from src.agentharm_gen.harness.dataset import load_generated_dataset
from src.agentharm_gen.harness.registry import build_grading_module
from src.agentharm_gen.harness.setup_tools import setup_generated_tools


@task
def agentharm_gen(
    tasks_path: str | None = None,
    n_irrelevant_tools: int = 0,
    agent: str = "default",
    agent_kwargs: dict | None = None,
    refusal_judge: str = "openai/gpt-4o-2024-08-06",
    semantic_judge: str = "openai/gpt-4o-2024-08-06",
) -> Task:
    """AgentHarm-Gen task backed by the generated tool+task+grader pipeline.

    Args:
        tasks_path: Path to tasks.jsonl (defaults to generated/tasks.jsonl).
        n_irrelevant_tools: Number of distractor tools to inject per sample.
        agent: Solver name (default: 'default' from inspect_evals agents).
        agent_kwargs: Extra kwargs forwarded to the agent solver.
        refusal_judge: Model ID for the refusal detection judge.
        semantic_judge: Model ID for the semantic task-completion judge.
    """
    dataset = load_generated_dataset(tasks_path)
    grading_module = build_grading_module()
    agent_solver = get_agent(agent=agent, **(agent_kwargs or {}))

    plan = [
        setup_generated_tools(n_irrelevant_tools=n_irrelevant_tools),
        agent_solver,
    ]

    return Task(
        dataset=dataset,
        solver=plan,
        scorer=combined_scorer(
            grading_module,
            RefusalJudgeLLM(judge_model=refusal_judge),
            get_model(semantic_judge),
        ),
        max_messages=20,
        config=GenerateConfig(
            max_retries=3,
            max_tokens=4096,
            temperature=0.0,
            max_connections=100,
        ),
    )
