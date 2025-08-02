import torch
import torch.nn as nn
from src.replay_buffer import ReplayBuffer
from typing import Optional, Dict, Any, List

class FixedHighLevelPolicy(nn.Module):
    """A trivial high-level policy that always returns the same instruction.

    This is useful for ablation experiments where we want to remove the
    hierarchical decision-making and study the effect of a *static* prompt
    provided to the low-level policy.
    """

    # Pretend to be a DQN-style high-level policy so that existing code paths
    # in *agent_trainer.py* work without modification.
    policy_kind: str = "dqn"

    def __init__(self, instruction: str = "Rephrase this", device: Optional[str] = None) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Store the single instruction as the only available strategy.
        self.strategies: List[Dict[str, Any]] = [
            {
                "id": "fixed_0",
                "instruction": instruction,
                "type": "fixed",
                "template": instruction,
            }
        ]

        # A tiny replay buffer is kept to satisfy calls to `replay_buffer.push`.
        self.replay_buffer = ReplayBuffer(capacity=1)

        # Dummy parameter so that `.parameters()` is not empty – required by the
        # optimizer creation logic in *main.py*.
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Move to device for completeness (even though we never compute tensors).
        self.to(self.device)

    # --------------------------------------------------------------------- API
    def select_action(self, *args, **kwargs) -> int:  # pylint: disable=unused-argument
        """Always return 0 – the index of the sole fixed strategy."""
        return 0

    # The forward pass is never used but is implemented for completeness.
    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        return torch.zeros(1, device=self.device)

    # ----------------------------------------------------------------- training
    def optimise_q_head(self) -> Dict[str, float]:
        """No-op optimisation step – returns empty metrics dict."""
        return {}

    # -------------------------------------------------------------- checkpointing
    def save(self, path: str) -> None:  # pylint: disable=unused-argument
        """Save nothing – fixed policy has no learnable state."""
        pass

    def load(self, path: str) -> None:  # pylint: disable=unused-argument
        """Load nothing – fixed policy has no learnable state."""
        pass 