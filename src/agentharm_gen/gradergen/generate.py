"""Component 3: Per-task grading function generator (A.3).

For each task record in tasks.jsonl, makes one LLM call to produce a
grade_{task_id}() async function, wraps it with the required imports,
validates it, and writes it to generated/graders/{task_id}.py.

Usage:
    from src.agentharm_gen.gradergen.generate import generate_all_graders
    generate_all_graders(tasks, out_dir="generated/graders")
"""

import ast
import json
import pathlib

from src.agentharm_gen.llm import call, extract_tag
from src.agentharm_gen.pipeline_status import (
    TASKS_PATH,
    assert_all_tasks_have_graders,
    print_pipeline_status,
    task_grader_is_complete,
    write_pipeline_status,
)
from src.agentharm_gen.validation import validate_grader_source
from src.agentharm_gen.gradergen.wrap import wrap_grader
from src.agentharm_gen.taskgen.generate import load_existing_tasks


GRADERS_DIR = pathlib.Path(__file__).parent.parent / "generated" / "graders"
REGISTRY_PATH = pathlib.Path(__file__).parent.parent / "generated" / "registry.json"
PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "gradergen_a3.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save_registry_graders(graders: dict[str, str]) -> None:
    data = _load_registry()
    data["graders"] = graders
    REGISTRY_PATH.write_text(json.dumps(data, indent=2))


def _smoke_test_grader(src: str, fn_name: str) -> None:
    """Verify the wrapped grader defines the expected async grade function."""
    tree = ast.parse(src)
    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == fn_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Grader module missing async function '{fn_name}'")


def generate_one_grader(
    task: dict,
    prompt_template: str,
    out_dir: pathlib.Path,
) -> str:
    """Generate, wrap, validate, and write one grader. Returns task_id on success."""
    task_id = task["id"]
    fn_name = task["grading_function"]
    out_path = out_dir / f"{task_id}.py"

    prompt = prompt_template.replace("{task}", task["prompt"])
    response = call(prompt, max_tokens=2000, temperature=1.0)
    try:
        raw_src = extract_tag(response, "grading_function")
    except ValueError as e:
        print(f"  [retry] grader parse failed for '{task_id}': {e}")
        response = call(prompt, max_tokens=2000, temperature=1.0)
        try:
            raw_src = extract_tag(response, "grading_function")
        except ValueError as e2:
            raise RuntimeError(
                f"Grader generation failed for task '{task_id}': {e2}"
            ) from e2

    full_src = wrap_grader(raw_src, task_id)

    validate_grader_source(full_src)
    _smoke_test_grader(full_src, fn_name)

    out_path.write_text(full_src)
    return task_id


def generate_all_graders(
    tasks: list[dict] | None = None,
    tasks_path: pathlib.Path | str | None = None,
    out_dir: pathlib.Path | str = GRADERS_DIR,
    resume: bool = True,
) -> dict[str, str]:
    """Generate graders for all tasks; return {task_id: grader_file_path} registry.

    Reads tasks from tasks_path if tasks is not provided.
    """
    if tasks is None:
        resolved_tasks_path = pathlib.Path(tasks_path) if tasks_path else (
            pathlib.Path(__file__).parent.parent / "generated" / "tasks.jsonl"
        )
        tasks = load_existing_tasks(resolved_tasks_path)
    else:
        resolved_tasks_path = pathlib.Path(tasks_path) if tasks_path else TASKS_PATH

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = _load_prompt()
    existing_data = _load_registry()
    existing_graders: dict[str, str] = existing_data.get("graders", {})
    graders = dict(existing_graders)

    print_pipeline_status()

    for task in tasks:
        task_id = task["id"]
        if resume and task_grader_is_complete(task, out_dir):
            print(f"  skipped (resume): {task_id}")
            continue
        generate_one_grader(task, prompt_template, out_dir)
        graders[task_id] = str(out_dir / f"{task_id}.py")
        _save_registry_graders(graders)
        print(f"  generated grader: {task_id}")

    assert_all_tasks_have_graders(resolved_tasks_path, out_dir)
    write_pipeline_status(resolved_tasks_path, out_dir)
    return graders
