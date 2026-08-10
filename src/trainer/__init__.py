"""
Trainer package exposing the legacy AgentTrainer API.

Re-export everything from `src.trainer.core` so existing imports can migrate
gradually.
"""

from .core import *  # noqa: F401,F403
from .rollout import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]

