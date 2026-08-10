"""Component 2: Adversarial task generator (A.2).

For each category, makes one LLM call to produce N task records
{task, target_functions}, validates them against the tool registry,
deduplicates, and appends to tasks.jsonl.

Usage:
    from src.agentharm_gen.taskgen.generate import generate_all_tasks
    generate_all_tasks(registry, n_per_call=5, out_path="generated/tasks.jsonl")
"""

import json
import pathlib
import re
import unicodedata
from typing import Any

from src.agentharm_gen.llm import call
from src.agentharm_gen.pipeline_status import (
    category_counts,
    load_tasks,
    print_pipeline_status,
    write_pipeline_status,
)
from src.agentharm_gen.taskgen.categories import CATEGORIES
from src.agentharm_gen.taskgen.parse import parse_task_records


TASKS_PATH = pathlib.Path(__file__).parent.parent / "generated" / "tasks.jsonl"
PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "taskgen_a2.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def _build_apps_json(registry: dict[str, str]) -> str:
    """Build the <apps> JSON payload from the tool registry.

    Each tool name encodes {method}_{app}. We reconstruct a list of
    {app, api, description} entries for the prompt.
    """
    entries = []
    for name in sorted(registry):
        # Split on last underscore to separate method from app name
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            api_name, app_name = parts
        else:
            api_name, app_name = name, "unknown"
        entries.append({"app": app_name, "api": api_name, "function": name})
    return json.dumps(entries, indent=2)


def _ngram_set(text: str, n: int = 3) -> set[str]:
    """Return the set of character n-grams from text (lowercased, normalized)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: str, b: str, n: int = 3) -> float:
    sa, sb = _ngram_set(a, n), _ngram_set(b, n)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _is_duplicate(task_text: str, existing_texts: list[str], threshold: float = 0.5) -> bool:
    for other in existing_texts:
        if _jaccard(task_text, other) > threshold:
            return True
    return False


def _make_id(category: str, index: int) -> str:
    slug = category.lower().replace(" ", "_")
    return f"{slug}-{index}"


def _validate_record(
    record: dict[str, Any],
    category: str,
    tool_registry: dict[str, str],
) -> None:
    """Raise ValueError if the record fails any validation check."""
    task_text = record["task"]
    tf = record["target_functions"]

    if not task_text:
        raise ValueError("Empty task text")
    if not tf:
        raise ValueError("Empty target_functions list")

    unknown = [fn for fn in tf if fn not in tool_registry]
    if unknown:
        raise ValueError(f"Unknown tools in target_functions: {unknown}")


def load_existing_tasks(path: pathlib.Path) -> list[dict]:
    """Load tasks from a JSONL file; return empty list if file absent."""
    return load_tasks(path)


def generate_for_category(
    category: str,
    tool_registry: dict[str, str],
    n: int,
    prompt_template: str,
    existing_tasks: list[dict],
    dedup_threshold: float = 0.5,
) -> list[dict]:
    """Generate and validate N task records for a single category.

    Returns only new, valid, non-duplicate records.
    """
    apps_json = _build_apps_json(tool_registry)
    prompt = (
        prompt_template.replace("{n}", str(n))
        .replace("{category}", category)
        .replace("{apps_json}", apps_json)
    )
    response = call(prompt, max_tokens=4000, temperature=1.0)

    try:
        raw_records = parse_task_records(response)
    except ValueError as e:
        print(f"  [retry] parse failed for '{category}': {e}")
        response = call(prompt, max_tokens=4000, temperature=1.0)
        try:
            raw_records = parse_task_records(response)
        except ValueError as e2:
            print(f"  [skip] category '{category}': {e2}")
            return []

    existing_in_category = [t for t in existing_tasks if t["category"] == category]
    accepted = []

    for rec in raw_records:
        try:
            _validate_record(rec, category, tool_registry)
        except ValueError as e:
            print(f"  [skip] validation failed: {e}")
            continue

        existing_texts = [t["prompt"] for t in existing_in_category] + [a["task"] for a in accepted]
        if _is_duplicate(rec["task"], existing_texts, dedup_threshold):
            print(f"  [skip] duplicate task: {rec['task'][:60]}...")
            continue

        accepted.append(rec)

    return accepted


def generate_all_tasks(
    tool_registry: dict[str, str],
    categories: list[str] | None = None,
    n_per_call: int = 5,
    out_path: pathlib.Path | str = TASKS_PATH,
    dedup_threshold: float = 0.5,
) -> list[dict]:
    """Generate tasks for all categories and append to tasks.jsonl.

    Returns the list of all newly accepted task records (not including pre-existing).
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    categories = categories or CATEGORIES
    prompt_template = _load_prompt()

    existing = load_existing_tasks(out_path)
    next_index = {cat: sum(1 for t in existing if t["category"] == cat)
                  for cat in CATEGORIES}

    all_new: list[dict] = []

    print_pipeline_status(out_path)

    with open(out_path, "a") as f:
        for category in categories:
            complete, pending = category_counts(category, existing)
            if pending > 0:
                print(
                    f"\nSkipping category {category}: "
                    f"{pending} tasks awaiting graders (not adding more)"
                )
                continue
            if complete >= n_per_call:
                print(
                    f"\nSkipping category {category}: "
                    f"{complete}/{n_per_call} tasks complete"
                )
                continue

            print(f"\nGenerating tasks for category: {category}")
            new_records = generate_for_category(
                category, tool_registry, n_per_call, prompt_template, existing, dedup_threshold
            )
            for rec in new_records:
                idx = next_index.get(category, 0)
                task_id = _make_id(category, idx)
                grading_fn = f"grade_{task_id.replace('-', '_')}"
                out_rec = {
                    "id": task_id,
                    "category": category,
                    "prompt": rec["task"],
                    "target_functions": rec["target_functions"],
                    "grading_function": grading_fn,
                }
                f.write(json.dumps(out_rec) + "\n")
                f.flush()
                existing.append(out_rec)
                all_new.append(out_rec)
                next_index[category] = idx + 1
                print(f"  accepted: {task_id} ({len(rec['target_functions'])} tools)")

    print(f"\nTotal new tasks: {len(all_new)}")
    write_pipeline_status(out_path)
    return all_new
