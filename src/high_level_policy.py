import copy
import math
import random
from typing import Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from template_registry import TemplateRegistry
from replay_buffer import ReplayBuffer


class TemplateHighLevelPolicy(nn.Module):
    """High-level policy that picks a template via DQN-style value estimation.

    The encoder (e.g. LLM) is provided externally. Only the small Q-head is trained by
    default to keep computation light.  This class purposefully keeps the interface
    minimal so that later we can swap it for a PPO-based LLM with the same public
    methods where possible.
    """

    policy_kind: str = "dqn"  # used by AgentTrainer to choose the correct update rule

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: Any,
        registry: TemplateRegistry,
        hidden_size: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        replay_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder  # may be frozen or fine-tuned by the caller
        self.tokenizer = tokenizer
        self.registry = registry
        self.num_actions = len(registry)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Simple two-layer head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_actions),
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Ensure Q-head is in bf16 for consistency
        self.q_head = self.q_head.to(dtype=torch.bfloat16)

        # Optimiser on the head only for now.
        self.optimizer = torch.optim.Adam(self.q_head.parameters(), lr=lr)

        # Epsilon-greedy parameters
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._steps_done = 0

        # Replay buffer and target network
        self.replay_buffer = ReplayBuffer(replay_size)
        self._target_q_head = copy.deepcopy(self.q_head).eval().to(self.device)
        # Ensure target Q-head is also in bf16
        self._target_q_head = self._target_q_head.to(dtype=torch.bfloat16)

    # ------------------------------------------------------------------ helpers
    def _current_epsilon(self) -> float:
        frac = math.exp(-1.0 * self._steps_done / self._epsilon_decay)
        return self._epsilon_end + (self._epsilon_start - self._epsilon_end) * frac

    def _encode_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return a pooled representation (last token) from the encoder."""
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state  # type: ignore[attr-defined]
            else:
                # Some models return a tuple; first item is hidden states
                hidden_states = outputs[0]  # type: ignore[index]
        # Use last token as pooled representation; alternative could be CLS
        return hidden_states[:, -1, :]

    # ------------------------------------------------------------------ API
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Q-values for every template in the registry."""
        pooled = self._encode_state(input_ids, attention_mask)
        return self.q_head(pooled)

    @torch.no_grad()
    def select_action(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, force_greedy: bool = False
    ) -> int:
        """Return a template index following ε-greedy exploration."""
        self._steps_done += 1
        epsilon = 0.0 if force_greedy else self._current_epsilon()
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        q_values = self.forward(input_ids.to(self.device), attention_mask.to(self.device) if attention_mask is not None else None)
        return int(torch.argmax(q_values, dim=-1).item())

    # ------------------------------------------------------------------ DQN update
    def optimise_q_head(self) -> dict[str, float]:
        """One DQN update step if enough data are in the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        transitions = self.replay_buffer.sample(self.batch_size)
        # Unpack
        state_texts, actions, rewards, next_state_texts, dones = zip(*transitions)

        # Tokenise & pad
        states_enc = self.tokenizer(list(state_texts), return_tensors="pt", padding=True, truncation=True)
        next_states_enc = self.tokenizer(list(next_state_texts), return_tensors="pt", padding=True, truncation=True)

        input_ids = states_enc.input_ids.to(self.device)
        attention_mask = states_enc.attention_mask.to(self.device)
        next_input_ids = next_states_enc.input_ids.to(self.device)
        next_attention_mask = next_states_enc.attention_mask.to(self.device)

        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.bfloat16, device=self.device)
        done_tensor = torch.tensor(dones, dtype=torch.bfloat16, device=self.device)

        # Q(s, a)
        q_values = self.forward(input_ids, attention_mask)
        state_action_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # target = r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self._target_q_head(self._encode_state(next_input_ids, next_attention_mask))
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_values = reward_tensor + self.gamma * max_next_q * (1 - done_tensor)

        loss = F.smooth_l1_loss(state_action_values, target_values)

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self._steps_done % self.target_update_freq == 0:
            self._target_q_head.load_state_dict(self.q_head.state_dict())

        return {"high_level_policy_loss": loss.item()}

    # ------------------------------------------------------------------ checkpoint helpers
    def save(self, path: str) -> None:
        torch.save({
            "q_head_state_dict": self.q_head.state_dict(),
            "target_state_dict": self._target_q_head.state_dict(),
            "steps_done": self._steps_done,
        }, path + ".pt")

    def load(self, path: str) -> None:
        ckpt = torch.load(path + ".pt", map_location=self.device)
        self.q_head.load_state_dict(ckpt["q_head_state_dict"])
        self._target_q_head.load_state_dict(ckpt["target_state_dict"])
        self._steps_done = ckpt["steps_done"] 