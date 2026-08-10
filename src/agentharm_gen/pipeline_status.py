"""Track which generated tasks are complete (task record + grader on disk)."""

import ast
import json
import pathlib
from typing import Any

from src.agentharm_gen.validation import validate_grader_source

GENERATED_DIR = pathlib.Path(__file__).parent / "generated"
TASKS_PATH = GENERATED_DIR / "tasks.jsonl"
GRADERS_DIR = GENERATED_DIR / "graders"
STATUS_PATH = GENERATED_DIR / "pipeline_status.json"


def grader_ids(graders_dir: pathlib.Path = GRADERS_DIR) -> set[str]:
    if not graders_dir.exists():
        return set()
    return {path.stem for path in graders_dir.glob("*.py")}


def task_grader_is_complete(
    task: dict[str, Any],
    graders_dir: pathlib.Path = GRADERS_DIR,
) -> bool:
    """True when graders/{task_id}.py exists and defines task['grading_function']."""
    path = graders_dir / f"{task['id']}.py"
    if not path.exists():
        return False
    src = path.read_text()
    fn_name = task["grading_function"]
    try:
        validate_grader_source(src)
        tree = ast.parse(src)
    except (ValueError, SyntaxError):
        return False
    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == fn_name
    ]
    return len(matches) == 1


def missing_grader_task_ids(
    tasks_path: pathlib.Path = TASKS_PATH,
    graders_dir: pathlib.Path = GRADERS_DIR,
) -> list[str]:
    tasks = load_tasks(tasks_path)
    return [task["id"] for task in tasks if not task_grader_is_complete(task, graders_dir)]


def assert_all_tasks_have_graders(
    tasks_path: pathlib.Path = TASKS_PATH,
    graders_dir: pathlib.Path = GRADERS_DIR,
) -> None:
    missing = missing_grader_task_ids(tasks_path, graders_dir)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise RuntimeError(
            f"{len(missing)} tasks in {tasks_path} are missing valid graders "
            f"under {graders_dir}: {preview}{suffix}. "
            "Run: python -m src.agentharm_gen.cli graders"
        )


def load_tasks(tasks_path: pathlib.Path = TASKS_PATH) -> list[dict[str, Any]]:
    if not tasks_path.exists():
        return []
    tasks: list[dict[str, Any]] = []
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def build_pipeline_status(
    tasks_path: pathlib.Path = TASKS_PATH,
    graders_dir: pathlib.Path = GRADERS_DIR,
) -> dict[str, Any]:
    tasks = load_tasks(tasks_path)
    completed = [task["id"] for task in tasks if task_grader_is_complete(task, graders_dir)]
    pending = [task["id"] for task in tasks if task["id"] not in completed]
    done = set(completed)

    by_category: dict[str, dict[str, Any]] = {}
    for task in tasks:
        category = task["category"]
        bucket = by_category.setdefault(
            category,
            {"complete": 0, "pending": 0, "complete_ids": [], "pending_ids": []},
        )
        if task["id"] in completed:
            bucket["complete"] += 1
            bucket["complete_ids"].append(task["id"])
        else:
            bucket["pending"] += 1
            bucket["pending_ids"].append(task["id"])

    return {
        "tasks_total": len(tasks),
        "graders_total": len(completed),
        "completed_task_ids": completed,
        "pending_task_ids": pending,
        "by_category": by_category,
    }


def write_pipeline_status(
    tasks_path: pathlib.Path = TASKS_PATH,
    graders_dir: pathlib.Path = GRADERS_DIR,
    status_path: pathlib.Path = STATUS_PATH,
) -> dict[str, Any]:
    status = build_pipeline_status(tasks_path, graders_dir)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2))
    return status


def print_pipeline_status(
    tasks_path: pathlib.Path = TASKS_PATH,
    graders_dir: pathlib.Path = GRADERS_DIR,
) -> dict[str, Any]:
    status = write_pipeline_status(tasks_path, graders_dir)
    print(
        f"Pipeline status: {len(status['completed_task_ids'])} complete, "
        f"{len(status['pending_task_ids'])} pending, "
        f"{status['tasks_total']} tasks in tasks.jsonl"
    )
    for category, counts in status["by_category"].items():
        print(
            f"  {category}: {counts['complete']} complete, {counts['pending']} pending"
        )
    if status["pending_task_ids"]:
        preview = ", ".join(status["pending_task_ids"][:8])
        suffix = "..." if len(status["pending_task_ids"]) > 8 else ""
        print(f"  pending ids: {preview}{suffix}")
    return status


def category_counts(
    category: str,
    tasks: list[dict[str, Any]],
    done: set[str] | None = None,
) -> tuple[int, int]:
    in_category = [task for task in tasks if task["category"] == category]
    complete = sum(1 for task in in_category if task_grader_is_complete(task))
    pending = sum(1 for task in in_category if not task_grader_is_complete(task))
    return complete, pending
