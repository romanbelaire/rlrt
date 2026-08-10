"""Load AppWorld standard API docs for the paper's app subset."""

import json
import pathlib
from typing import Any

APP_SUBSET = ["venmo", "spotify", "phone", "amazon", "gmail"]


def find_appworld_docs_dir() -> pathlib.Path:
    """Find AppWorld's standard API docs directory.

    Checks APPWORLD_DATA_DIR, then APPWORLD_ROOT/data, then the installed
    package data path.
    """
    import os

    candidates: list[pathlib.Path] = []

    env_data_dir = os.environ.get("APPWORLD_DATA_DIR")
    if env_data_dir:
        candidates.append(pathlib.Path(env_data_dir) / "api_docs" / "standard")

    appworld_root = os.environ.get("APPWORLD_ROOT", os.getcwd())
    candidates.append(pathlib.Path(appworld_root) / "data" / "api_docs" / "standard")

    try:
        import appworld
        pkg_root = pathlib.Path(appworld.__file__).parent
        candidates.append(pkg_root / "data" / "api_docs" / "standard")
    except ImportError:
        pass

    for p in candidates:
        if p.exists():
            return p

    raise RuntimeError(
        "AppWorld API docs not found. After 'pip install appworld', run:\n"
        "  appworld install\n"
        "  appworld download data --root <APPWORLD_ROOT>\n"
        "Then set APPWORLD_DATA_DIR=<APPWORLD_ROOT>/data"
    )


def load_app_apis(app: str, docs_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Load all API entries for a single app from AppWorld standard JSON."""
    # AppWorld stores each app as {app}.json (e.g. venmo.json)
    path = docs_dir / f"{app}.json"
    if not path.exists():
        raise FileNotFoundError(f"No API doc file for app '{app}' at {path}")
    with open(path) as f:
        data = json.load(f)
    # Standard format: list, {"apis": [...]}, or {api_name: api_dict, ...}
    if isinstance(data, list):
        return data
    if "apis" in data:
        return data["apis"]
    if all(isinstance(v, dict) for v in data.values()):
        return list(data.values())
    raise ValueError(f"Unexpected AppWorld JSON structure in {path}")


def load_all_apis(
    apps: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {app: [api_dict, ...]} for each app in the subset."""
    apps = apps or APP_SUBSET
    docs_dir = find_appworld_docs_dir()
    return {app: load_app_apis(app, docs_dir) for app in apps}


def load_browser_apis() -> list[dict[str, Any]]:
    """Load the hand-authored browser API definitions."""
    browser_path = pathlib.Path(__file__).parent / "browser_apis.json"
    with open(browser_path) as f:
        return json.load(f)
