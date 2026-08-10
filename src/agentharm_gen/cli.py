"""CLI orchestrator for the AgentHarm-Gen data-generation pipeline.

Commands:
    convert  -- Generate @tool Python files from AppWorld + browser APIs
    tasks    -- Generate adversarial task records (tasks.jsonl)
    graders  -- Generate per-task grading functions
    eval     -- Run the generated benchmark via inspect eval
    refine   -- Run the iterative prompt refinement loop (Algorithm 1)

Pipeline order: convert → tasks → graders → eval
"""

import argparse
import asyncio
import pathlib
import sys


def cmd_convert(args: argparse.Namespace) -> None:
    """Convert AppWorld + browser APIs to @tool Python files."""
    from src.agentharm_gen.converter.appworld_loader import load_all_apis, load_browser_apis
    from src.agentharm_gen.converter.convert import build_toolset

    apps = args.apps.split(",") if args.apps else None
    app_docs = load_all_apis(apps)
    browser_apis = load_browser_apis()

    print(f"Converting {sum(len(v) for v in app_docs.values())} AppWorld + {len(browser_apis)} browser APIs...")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else None
    kwargs = {}
    if out_dir:
        kwargs["out_dir"] = out_dir

    registry = build_toolset(
        app_docs=app_docs,
        browser_apis=browser_apis,
        resume=not args.no_resume,
        **kwargs,
    )
    print(f"\nDone. {len(registry)} tools in registry.")


def cmd_tasks(args: argparse.Namespace) -> None:
    """Generate adversarial task records for all categories."""
    import json
    import pathlib
    from src.agentharm_gen.taskgen.generate import generate_all_tasks

    registry_path = pathlib.Path(__file__).parent / "generated" / "registry.json"
    if not registry_path.exists():
        print("ERROR: No registry.json found. Run 'convert' first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(registry_path.read_text())
    tool_registry = data.get("tools", {})
    if not tool_registry:
        print("ERROR: No tools found in registry. Run 'convert' first.", file=sys.stderr)
        sys.exit(1)

    categories = args.categories.split(",") if args.categories else None
    out_path = pathlib.Path(args.out_path) if args.out_path else None

    kwargs = {}
    if out_path:
        kwargs["out_path"] = out_path

    generate_all_tasks(
        tool_registry=tool_registry,
        categories=categories,
        n_per_call=args.n,
        **kwargs,
    )


def cmd_graders(args: argparse.Namespace) -> None:
    """Generate grading functions for all tasks in tasks.jsonl."""
    from src.agentharm_gen.gradergen.generate import generate_all_graders

    tasks_path = pathlib.Path(args.tasks_path) if args.tasks_path else None
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else None

    kwargs = {}
    if tasks_path:
        kwargs["tasks_path"] = tasks_path
    if out_dir:
        kwargs["out_dir"] = out_dir

    try:
        graders = generate_all_graders(resume=not args.no_resume, **kwargs)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone. {len(graders)} graders in registry.")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run the generated benchmark via inspect eval (delegates to inspect CLI)."""
    import subprocess

    task_path = "src/agentharm_gen/harness/task.py@agentharm_gen"
    cmd = [
        "inspect", "eval", task_path,
        "--model", args.model,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.semantic_judge:
        cmd += ["-T", f"semantic_judge={args.semantic_judge}"]
    if args.refusal_judge:
        cmd += ["-T", f"refusal_judge={args.refusal_judge}"]
    if args.tasks_path:
        cmd += ["-T", f"tasks_path={args.tasks_path}"]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_registry(args: argparse.Namespace) -> None:
    """Rebuild tool_registry.json from existing generated/tools/*.py files."""
    from src.agentharm_gen.harness.registry import write_tool_registry_from_dir

    tools_dir = pathlib.Path(args.tools_dir) if args.tools_dir else None
    ordered = write_tool_registry_from_dir(tools_dir)
    print(f"Wrote tool_registry.json with {len(ordered)} tools.")


def cmd_refine(args: argparse.Namespace) -> None:
    """Run iterative prompt refinement (Algorithm 1) — research seam."""
    print(
        "The 'refine' command is the research contribution boundary.\n"
        "Implement your AttackerPolicy in src/agentharm_gen/optimizer/attacker.py,\n"
        "then wire it to run_agent + grade from the harness.\n"
        "See optimizer/refine.py for the full Algorithm 1 skeleton."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.agentharm_gen.cli",
        description="AgentHarm-Gen data-generation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # convert
    p_conv = sub.add_parser("convert", help="Generate @tool files from API docs")
    p_conv.add_argument("--apps", default=None,
                        help="Comma-separated app subset (default: venmo,spotify,phone,amazon,gmail)")
    p_conv.add_argument("--out-dir", dest="out_dir", default=None,
                        help="Output directory for generated tool .py files")
    p_conv.add_argument("--no-resume", action="store_true",
                        help="Regenerate all tools even if already in registry")

    # tasks
    p_tasks = sub.add_parser("tasks", help="Generate adversarial task records")
    p_tasks.add_argument("--categories", default=None,
                         help="Comma-separated category subset (default: all 8)")
    p_tasks.add_argument("-n", type=int, default=5,
                         help="Number of tasks to generate per LLM call per category")
    p_tasks.add_argument("--out-path", dest="out_path", default=None,
                         help="Output path for tasks.jsonl")

    # graders
    p_grad = sub.add_parser("graders", help="Generate per-task grading functions")
    p_grad.add_argument("--tasks-path", dest="tasks_path", default=None,
                        help="Path to tasks.jsonl")
    p_grad.add_argument("--out-dir", dest="out_dir", default=None,
                        help="Output directory for grader .py files")
    p_grad.add_argument("--no-resume", action="store_true",
                        help="Regenerate all graders even if already done")

    # eval
    p_eval = sub.add_parser("eval", help="Run inspect eval on generated benchmark")
    p_eval.add_argument("--model", required=True, help="Target model (e.g. openai/gpt-4o)")
    p_eval.add_argument("--limit", type=int, default=None, help="Max samples to eval")
    p_eval.add_argument("--semantic-judge", dest="semantic_judge", default=None)
    p_eval.add_argument("--refusal-judge", dest="refusal_judge", default=None)
    p_eval.add_argument("--tasks-path", dest="tasks_path", default=None)

    # registry
    p_reg = sub.add_parser("registry", help="Rebuild tool_registry.json from generated/tools/")
    p_reg.add_argument("--tools-dir", dest="tools_dir", default=None)

    # refine
    sub.add_parser("refine", help="Iterative prompt refinement (Algorithm 1 skeleton)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "convert": cmd_convert,
        "tasks": cmd_tasks,
        "graders": cmd_graders,
        "eval": cmd_eval,
        "registry": cmd_registry,
        "refine": cmd_refine,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
