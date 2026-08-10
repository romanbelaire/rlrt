"""Load generated tasks.jsonl as inspect_ai Samples.

The record schema matches the official AgentHarm format expected by combined_scorer:
  {id, category, prompt, target_functions, grading_function}
"""

import json
import pathlib

from inspect_ai.dataset import Dataset, Sample, FieldSpec
from inspect_evals.utils import load_json_dataset

from src.agentharm_gen.pipeline_status import assert_all_tasks_have_graders


TASKS_PATH = pathlib.Path(__file__).parent.parent / "generated" / "tasks.jsonl"


def load_generated_dataset(path: pathlib.Path | str | None = None) -> Dataset:
    """Load generated tasks.jsonl as an inspect_ai Dataset.

    The Sample schema matches the official AgentHarm format:
      - input = prompt
      - id = id
      - metadata: id, category, prompt, target_functions, grading_function
    """
    tasks_path = pathlib.Path(path) if path else TASKS_PATH
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"Generated tasks not found at {tasks_path}. "
            "Run: python -m src.agentharm_gen.cli tasks"
        )

    assert_all_tasks_have_graders(tasks_path)

    return load_json_dataset(
        str(tasks_path),
        "agentharm_gen",
        sample_fields=FieldSpec(
            input="prompt",
            id="id",
            metadata=[
                "id",
                "category",
                "prompt",
                "target_functions",
                "grading_function",
            ],
        ),
    )
