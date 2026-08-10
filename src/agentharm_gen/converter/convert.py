"""Component 1: API doc → @tool Python function generator.

Converts AppWorld standard-format API entries (plus hand-authored browser APIs)
into inspect_ai @tool-decorated async functions and writes them to generated/tools/.

Usage:
    from src.agentharm_gen.converter.convert import build_toolset
    registry = build_toolset(out_dir="src/agentharm_gen/generated/tools")
"""

import json
import pathlib
import re
from typing import Any

from src.agentharm_gen.llm import call, extract_tag
from src.agentharm_gen.validation import validate_tool_source
from src.agentharm_gen.converter.api_doc_to_text import appworld_api_to_text, browser_api_to_text, tool_name


GENERATED_TOOLS_DIR = pathlib.Path(__file__).parent.parent / "generated" / "tools"
REGISTRY_PATH = pathlib.Path(__file__).parent.parent / "generated" / "registry.json"
PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "converter_a1.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def _normalize_inner_fn(src: str) -> str:
    """Rename inner async function from 'run' to 'execute' if needed."""
    return re.sub(r"\basync def run\b", "async def execute", src)


def _convert_one(api_doc_text: str, prompt_template: str) -> str:
    """Single LLM call: api doc text → validated @tool source."""
    prompt = prompt_template.replace("{api_docs}", api_doc_text)
    response = call(prompt, max_tokens=2000, temperature=1.0)
    src = extract_tag(response, "function")
    src = _normalize_inner_fn(src)
    validate_tool_source(src)
    return src


def _load_registry(out_dir: pathlib.Path) -> dict[str, str]:
    registry_path = out_dir.parent / "registry.json"
    if registry_path.exists():
        data = json.loads(registry_path.read_text())
        return data.get("tools", {})
    return {}


def _save_registry(registry: dict[str, str], out_dir: pathlib.Path) -> None:
    registry_path = out_dir.parent / "registry.json"
    existing: dict[str, Any] = {}
    if registry_path.exists():
        existing = json.loads(registry_path.read_text())
    existing["tools"] = registry
    registry_path.write_text(json.dumps(existing, indent=2))


def convert_one_api(
    app: str,
    api: dict[str, Any],
    out_dir: pathlib.Path,
    prompt_template: str,
    existing_registry: dict[str, str],
    resume: bool = True,
) -> tuple[str, str] | None:
    """Convert a single API doc to a @tool file.

    Returns (name, src) on success, or None if skipped (resume=True and already done).
    Raises on validation failure.
    """
    name = tool_name(app, api)
    if resume and name in existing_registry:
        return None

    if app == "browser":
        doc_text = browser_api_to_text(api)
    else:
        doc_text = appworld_api_to_text(app, api)

    src = _convert_one(doc_text, prompt_template)
    out_path = out_dir / f"{name}.py"
    out_path.write_text(src)
    return name, src


def build_toolset(
    app_docs: dict[str, list[dict[str, Any]]],
    browser_apis: list[dict[str, Any]] | None = None,
    out_dir: pathlib.Path | str = GENERATED_TOOLS_DIR,
    resume: bool = True,
) -> dict[str, str]:
    """Convert all APIs to @tool files; return {name: file_path} registry.

    Args:
        app_docs: {app: [api_dict, ...]} from AppWorld loader.
        browser_apis: list of browser API dicts (loaded from browser_apis.json).
        out_dir: directory to write generated .py files.
        resume: skip APIs already present in registry.json.

    Returns:
        dict mapping tool name → absolute file path (all successfully validated tools).
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = _load_prompt()
    existing_registry = _load_registry(out_dir) if resume else {}
    registry = dict(existing_registry)

    all_apis: list[tuple[str, dict[str, Any]]] = []
    for app, apis in app_docs.items():
        for api in apis:
            all_apis.append((app, api))
    if browser_apis:
        for api in browser_apis:
            all_apis.append(("browser", api))

    for app, api in all_apis:
        name = tool_name(app, api)
        result = convert_one_api(app, api, out_dir, prompt_template, registry, resume)
        if result is not None:
            tool_n, _ = result
            registry[tool_n] = str(out_dir / f"{tool_n}.py")
            _save_registry(registry, out_dir)
            print(f"  converted: {tool_n}")
        else:
            print(f"  skipped (resume): {name}")

    from src.agentharm_gen.harness.registry import write_tool_registry_from_dir

    write_tool_registry_from_dir(out_dir)
    return registry
