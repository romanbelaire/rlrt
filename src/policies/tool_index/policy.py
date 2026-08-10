"""Layer 1: MLP tool-index policy over attacker encoder hidden states."""

import copy
import json
import math
import os
import random
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import disable_lora_adapters, get_transformer_module

from src.replay_buffer import ReplayBuffer


class ToolIndexPolicy(nn.Module):
    """Predict which environment tool the defender should call next."""

    policy_kind: str = "tool_index"

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: Any,
        tool_registry: List[str],
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 350,
        replay_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        max_chunk_size: int = 2,
        max_seq_length: int = 2048,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.tool_registry = list(tool_registry)
        self.num_actions = len(self.tool_registry)
        if self.num_actions == 0:
            raise ValueError("tool_registry must not be empty")

        actual_hidden_size = encoder.config.hidden_size
        self.hidden_size = actual_hidden_size
        print(f"[TOOL-INDEX-POLICY] registry_size={self.num_actions} hidden_size={actual_hidden_size}")

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_chunk_size = max_chunk_size
        self.max_seq_length = max_seq_length

        self.q_head = nn.Sequential(
            nn.Linear(actual_hidden_size, actual_hidden_size),
            nn.ReLU(),
            nn.Linear(actual_hidden_size, self.num_actions),
        )

        self.encoder_device = encoder.get_input_embeddings().weight.device
        self.device = self.encoder_device
        self.to(self.device)
        self.q_head = self.q_head.to(dtype=torch.bfloat16)

        self.optimizer = torch.optim.Adam(self.q_head.parameters(), lr=lr)

        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._steps_done = 0

        self.replay_buffer = ReplayBuffer(replay_size)
        self._target_q_head = copy.deepcopy(self.q_head).eval().to(self.device)
        self._target_q_head = self._target_q_head.to(dtype=torch.bfloat16)

    def get_tool_name(self, index: int) -> str:
        return self.tool_registry[index]

    def _current_epsilon(self) -> float:
        frac = math.exp(-1.0 * self._steps_done / self._epsilon_decay)
        return self._epsilon_end + (self._epsilon_start - self._epsilon_end) * frac

    def _transformer(self) -> nn.Module:
        """Return the frozen transformer backbone (skip LM head)."""
        return get_transformer_module(self.encoder)

    def _run_transformer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        with disable_lora_adapters(self.encoder):
            return self._transformer()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

    def _tokenize_batch(self, texts: List[str]) -> Any:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.max_seq_length,
        )

    def _resolve_chunk_size(self, batch_size: int, seq_len: int) -> int:
        if torch.cuda.is_available():
            from src.utils import get_optimal_chunk_size

            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            free_gb = (total_memory - allocated) / (1024**3)
            optimal = get_optimal_chunk_size(
                batch_size, free_gb * 0.6, seq_len, model_size_gb=8.0
            )
            return max(1, min(optimal, self.max_chunk_size))
        return max(1, min(batch_size, self.max_chunk_size))

    def _encode_state(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        target_device = self.encoder.get_input_embeddings().weight.device
        if input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)

        head_dtype = next(self.q_head.parameters()).dtype
        with torch.no_grad():
            outputs = self._run_transformer(input_ids, attention_mask)
            hidden_states = outputs.last_hidden_state

        return hidden_states[:, -1, :].to(dtype=head_dtype)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pooled = self._encode_state(input_ids, attention_mask)
        head_device = next(self.q_head.parameters()).device
        if pooled.device != head_device:
            pooled = pooled.to(head_device)
        return self.q_head(pooled)

    @torch.no_grad()
    def select_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        force_greedy: bool = False,
    ) -> int:
        self._steps_done += 1
        epsilon = 0.0 if force_greedy else self._current_epsilon()
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        q_values = self.forward(input_ids, attention_mask)
        return int(torch.argmax(q_values, dim=-1).item())

    def act(
        self,
        state_texts: List[str],
    ) -> Tuple[List[int], List[str], List[str]]:
        """Return tool indices, tool names, and state texts for replay."""
        indices: List[int] = []
        names: List[str] = []
        for state_text in state_texts:
            enc = self._tokenize_batch([state_text])
            idx = self.select_action(enc.input_ids, enc.attention_mask)
            indices.append(idx)
            names.append(self.get_tool_name(idx))
        return indices, names, state_texts

    def optimise_q_head(self) -> dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        transitions = self.replay_buffer.sample(self.batch_size)
        state_texts, actions, rewards, next_state_texts, dones = zip(*transitions)

        states_enc = self._tokenize_batch(list(state_texts))
        next_states_enc = self._tokenize_batch(list(next_state_texts))

        input_ids = states_enc.input_ids
        attention_mask = states_enc.attention_mask
        next_input_ids = next_states_enc.input_ids
        next_attention_mask = next_states_enc.attention_mask

        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.bfloat16, device=self.device)
        done_tensor = torch.tensor(dones, dtype=torch.bfloat16, device=self.device)

        batch_size = action_tensor.size(0)
        seq_len = input_ids.size(1)
        chunk_size = self._resolve_chunk_size(batch_size, seq_len)
        num_chunks = (batch_size + chunk_size - 1) // chunk_size

        print(
            f"Using chunked tool-index DQN: batch_size={batch_size}, "
            f"seq_len={seq_len}, chunk_size={chunk_size}"
        )

        double_next_q_parts: List[torch.Tensor] = []
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            with torch.no_grad():
                next_pooled = self._encode_state(
                    next_input_ids[chunk_start:chunk_end],
                    next_attention_mask[chunk_start:chunk_end],
                )
                # Double DQN: argmax online Q, evaluate with target Q
                next_online_q = self.q_head(next_pooled)
                best_actions = torch.argmax(next_online_q, dim=1)
                next_target_q = self._target_q_head(next_pooled)
                double_next_q_parts.append(
                    next_target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                )
            del next_pooled, next_online_q, next_target_q, best_actions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        double_next_q = torch.cat(double_next_q_parts, dim=0)
        target_values = reward_tensor + self.gamma * double_next_q * (1 - done_tensor)
        del double_next_q_parts, double_next_q

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            q_values = self.forward(
                input_ids[chunk_start:chunk_end],
                attention_mask[chunk_start:chunk_end],
            )
            state_action_values = q_values.gather(
                1, action_tensor[chunk_start:chunk_end].unsqueeze(1)
            ).squeeze(1)
            chunk_loss = F.smooth_l1_loss(
                state_action_values, target_values[chunk_start:chunk_end]
            )
            (chunk_loss / num_chunks).backward()
            total_loss += chunk_loss.item()
            del q_values, state_action_values, chunk_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.optimizer.step()

        if self._steps_done % self.target_update_freq == 0:
            self._target_q_head.load_state_dict(self.q_head.state_dict())

        del states_enc, next_states_enc, input_ids, attention_mask
        del next_input_ids, next_attention_mask, target_values
        del action_tensor, reward_tensor, done_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"tool_index_policy_loss": total_loss / num_chunks}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "q_head_state_dict": self.q_head.state_dict(),
                "target_state_dict": self._target_q_head.state_dict(),
                "steps_done": self._steps_done,
            },
            os.path.join(path, "model.pt"),
        )
        meta = {
            "policy_type": "tool_index",
            "policy_kind": self.policy_kind,
            "num_actions": self.num_actions,
            "tool_registry": self.tool_registry,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "max_chunk_size": self.max_chunk_size,
            "max_seq_length": self.max_seq_length,
            "epsilon_start": self._epsilon_start,
            "epsilon_end": self._epsilon_end,
            "epsilon_decay": self._epsilon_decay,
            "hidden_size": self.hidden_size,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
