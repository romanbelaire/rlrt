"""Scoring for the generated AgentHarm-Gen task.

Reuses the official combined_scorer from inspect_evals with a dynamically
built grading module instead of the static harmful_grading_functions module.
"""

from inspect_ai.model import get_model
from inspect_ai.scorer import Scorer

from inspect_evals.agentharm.scorer import RefusalJudgeLLM, combined_scorer

from src.agentharm_gen.harness.registry import build_grading_module


def make_scorer(
    refusal_judge_model: str = "openai/gpt-4o-2024-08-06",
    semantic_judge_model: str = "openai/gpt-4o-2024-08-06",
) -> Scorer:
    """Build a scorer backed by the generated grading module.

    Reuses the official combined_scorer which does:
      1. getattr(grading_module, metadata['grading_function'])(metadata, messages)
      2. RefusalJudgeLLM check on the assistant's final message

    The grading_module here is dynamically built from generated/graders/*.py.
    """
    grading_module = build_grading_module()
    refusal_judge = RefusalJudgeLLM(judge_model=refusal_judge_model)
    semantic_judge = get_model(semantic_judge_model)
    return combined_scorer(grading_module, refusal_judge, semantic_judge)
