"""Parse <task> + <target_functions> pairs from LLM output."""

import json
import re


def _extract_tag_blocks(text: str, tag: str) -> list[str]:
    """Extract <tag>...</tag> blocks; tolerate missing closing tags."""
    closed = re.compile(rf"<{tag}>(.*?)</{tag}>", re.S)
    blocks = [m.group(1).strip() for m in closed.finditer(text)]
    if blocks:
        return blocks

    open_blocks = re.compile(rf"<{tag}>(.*?)(?=<[a-zA-Z_][a-zA-Z0-9_]*>|$)", re.S)
    return [m.group(1).strip() for m in open_blocks.finditer(text)]


def _find_string_list_arrays(text: str) -> list[list[str]]:
    """Find JSON arrays of tool-name strings anywhere in the LLM output."""
    decoder = json.JSONDecoder()
    arrays: list[list[str]] = []
    idx = 0
    while idx < len(text):
        if text[idx] != "[":
            idx += 1
            continue
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            arrays.append(value)
        idx = end
    return arrays


def _parse_target_functions(raw: str) -> list[str]:
    try:
        tf_list = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse target_functions JSON: {raw!r}") from e
    if not isinstance(tf_list, list) or not all(isinstance(x, str) for x in tf_list):
        raise ValueError(f"target_functions must be a list of strings, got: {tf_list!r}")
    return tf_list


def parse_task_records(text: str) -> list[dict]:
    """Extract all (task, target_functions) pairs from LLM output.

    Expects blocks of the form:
        <task>...</task>
        <target_functions>["fn1", "fn2", ...]</target_functions>

    Falls back to bare JSON string arrays when <target_functions> tags are absent.
    Returns a list of dicts with keys 'task' and 'target_functions'.
    Raises ValueError if no valid pairs are found.
    """
    tasks = _extract_tag_blocks(text, "task")
    tfs_raw = _extract_tag_blocks(text, "target_functions")

    if not tasks:
        raise ValueError(f"No <task> blocks found in LLM output:\n{text[:500]}")

    tf_lists: list[list[str]] | None = None
    if len(tfs_raw) == len(tasks):
        tf_lists = [_parse_target_functions(raw) for raw in tfs_raw]
    elif not tfs_raw:
        arrays = _find_string_list_arrays(text)
        if len(arrays) == len(tasks):
            tf_lists = arrays

    if tf_lists is None:
        raise ValueError(
            f"Mismatch: {len(tasks)} <task> blocks vs {len(tfs_raw)} <target_functions> blocks"
        )

    return [{"task": task_text, "target_functions": tf_list}
            for task_text, tf_list in zip(tasks, tf_lists)]
