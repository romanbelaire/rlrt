"""Wrap raw LLM-generated grader body with required imports and rename the function."""

GRADER_HEADER = """\
from typing import Any, Dict, List

from inspect_ai.model import ChatMessage

from inspect_evals.agentharm.benchmark.grading_utils import (
    is_function_called,
    is_correct_order,
    extract_last_text_message,
    semantic_judge,
)

"""


def wrap_grader(raw_fn_src: str, task_id: str) -> str:
    """Prepend fixed imports to raw grader source and rename to grade_{task_id}.

    Args:
        raw_fn_src: The raw grader function source extracted from the LLM output.
                    Should contain 'async def grade_fictitious_task(...)'.
        task_id: The task record id (e.g. 'copyright_3').

    Returns:
        Complete, importable grader module source.
    """
    fn_name = f"grade_{task_id.replace('-', '_')}"
    renamed = raw_fn_src.replace(
        "async def grade_fictitious_task(", f"async def {fn_name}("
    )
    return GRADER_HEADER + renamed
