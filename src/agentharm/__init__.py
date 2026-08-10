"""AgentHarm red-team environment (lazy imports — see env.py / factory.py)."""

__all__ = ["AgentHarmEnvironment", "build_agentharm_environment"]


def __getattr__(name: str):
    if name == "AgentHarmEnvironment":
        from src.agentharm.env import AgentHarmEnvironment

        return AgentHarmEnvironment
    if name == "build_agentharm_environment":
        from src.agentharm.factory import build_agentharm_environment

        return build_agentharm_environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
