"""Dynamic tool and grader registries for AgentHarm-Gen.

Loads generated artifacts at runtime from generated/tools/ and generated/graders/,
building structures compatible with the existing inspect_evals scoring and tool-injection
patterns.
"""

import json
import pathlib
import types

from src.agentharm_gen.validation import (
    load_grader_from_source,
    load_tool_from_source,
    validate_grader_source,
    validate_tool_source,
)

GENERATED_DIR = pathlib.Path(__file__).parent.parent / "generated"
TOOLS_DIR = GENERATED_DIR / "tools"
GRADERS_DIR = GENERATED_DIR / "graders"
REGISTRY_PATH = GENERATED_DIR / "registry.json"
TOOL_REGISTRY_PATH = GENERATED_DIR / "tool_registry.json"


def write_tool_registry_from_dir(tools_dir: pathlib.Path | None = None) -> list[str]:
    """Rebuild tool_registry.json from existing generated/tools/*.py files."""
    tools_dir = tools_dir or TOOLS_DIR
    if not tools_dir.exists():
        raise FileNotFoundError(f"Generated tools directory not found: {tools_dir}")
    names = sorted(p.stem for p in tools_dir.glob("*.py"))
    if not names:
        raise ValueError(f"No tool .py files found in {tools_dir}")
    TOOL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOOL_REGISTRY_PATH.write_text(json.dumps({"tools": names}, indent=2))
    return names


def load_tool_registry(path: pathlib.Path | str | None = None) -> list[str]:
    """Load ordered tool names for L1 action indices."""
    registry_path = pathlib.Path(path) if path else TOOL_REGISTRY_PATH
    if not registry_path.exists():
        return write_tool_registry_from_dir(TOOLS_DIR)
    data = json.loads(registry_path.read_text())
    tools = data["tools"]
    return tools


def _read_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text())


def load_tool(name: str) -> object:
    """Load and return the @tool factory callable for the given tool name.

    Validates the source before loading. The returned callable is the outer
    @tool-decorated function (call it to get a Tool instance for inject into inspect_ai).
    """
    path = TOOLS_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Generated tool not found: {path}")
    src = path.read_text()
    validate_tool_source(src)
    mod = load_tool_from_source(src, f"agentharm_gen.generated.tools.{name}")
    # The @tool factory is the only top-level function defined in the module
    candidates = [
        v for k, v in mod.__dict__.items()
        if callable(v) and not k.startswith("_") and k not in ("tool", "Tool")
    ]
    if not candidates:
        raise ValueError(f"No callable found in generated tool module: {name}")
    return candidates[0]


def load_all_tools() -> dict[str, object]:
    """Load all @tool factories from generated/tools/. Returns {name: factory}."""
    if not TOOLS_DIR.exists():
        return {}
    tools = {}
    for py_file in sorted(TOOLS_DIR.glob("*.py")):
        name = py_file.stem
        try:
            tools[name] = load_tool(name)
        except Exception as e:
            raise RuntimeError(f"Failed to load generated tool '{name}': {e}") from e
    return tools


def build_grading_module() -> types.ModuleType:
    """Build a synthetic module containing all grade_{task_id} functions.

    The combined_scorer from inspect_evals does:
        await getattr(grading_module, grading_function_name)(metadata, messages)

    This synthetic module satisfies that contract by collecting every
    grading function from generated/graders/*.py as an attribute.
    """
    if not GRADERS_DIR.exists():
        raise FileNotFoundError(f"Generated graders directory not found: {GRADERS_DIR}")

    module = types.ModuleType("agentharm_gen.generated_grading_functions")

    for py_file in sorted(GRADERS_DIR.glob("*.py")):
        src = py_file.read_text()
        validate_grader_source(src)
        grader_mod = load_grader_from_source(src, f"agentharm_gen.grader.{py_file.stem}")
        # Collect all async functions that start with grade_
        for attr_name in dir(grader_mod):
            if attr_name.startswith("grade_"):
                fn = getattr(grader_mod, attr_name)
                setattr(module, attr_name, fn)

    return module
