from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from src.base import Environment


class LMRLGymEnvAdapter(Environment):
    """
    Adapter to use LMRL-Gym TextEnv/BatchedTextEnv within our Environment interface.

    This adapter maintains a batched text history per slot, converts our action
    strings to LMRL `Text` actions, and exposes step/reset consistent with
    our trainer expectations.

    Fail-fast behavior is enforced per repository policy.
    """

    def __init__(
        self,
        lmrl_env: Any,
        batch_size: int = 1,
        task: str = "maze",
        task_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from LLM_RL.environment import Text, TextHistory, TextEnv, BatchedTextEnv, TextEnvToBatchedTextEnv
        except Exception as exc:
            raise ImportError(
                "Failed to import LMRL-Gym core. Ensure the repo is installed with: "
                "pip install -e /path/to/LMRL-Gym --no-deps"
            ) from exc

        self.Text = Text
        self.TextHistory = TextHistory
        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.task = task
        self.task_options = task_options or {}

        # Normalize env to batched form
        if hasattr(lmrl_env, "step") and hasattr(lmrl_env, "reset") and not hasattr(lmrl_env, "copy"):
            # Likely TextEnv (not Batched)
            self.env = TextEnvToBatchedTextEnv(lmrl_env)
        else:
            # Assume already batched
            self.env = lmrl_env

        # Internal per-batch text histories
        self._histories: Optional[List[Optional[Tuple[self.Text, ...]]]] = None
        self._dones: Optional[List[bool]] = None

    # --- Environment API ---
    def reset(self) -> List[List[str]]:
        seed_list = [self.task_options.get("seed")] * self._batch_size
        options_list = [self.task_options.get("env_options")] * self._batch_size
        self._histories = self.env.reset(seed=seed_list, options=options_list)
        self._dones = [False] * self._batch_size
        return [self._history_to_strings(h) for h in self._histories]

    def step(
        self,
        actions: Union[List[str], str],
        **kwargs,
    ) -> Tuple[List[List[str]], List[str], torch.Tensor, torch.Tensor]:
        if isinstance(actions, str):
            actions = [actions]
        if not isinstance(actions, list) or len(actions) != self._batch_size:
            raise ValueError(
                f"actions must be a list[str] of length {self._batch_size}, got {type(actions)} with len={len(actions) if isinstance(actions, list) else 'NA'}"
            )
        if self._histories is None or self._dones is None:
            raise RuntimeError("reset() must be called before step()")

        # Build action-appended histories
        appended_histories: List[Optional[Tuple[self.Text, ...]]] = []
        for idx, (hist, done_flag, action_text) in enumerate(zip(self._histories, self._dones, actions)):
            if done_flag:
                appended_histories.append(None)
                continue
            parsed_action = self._parse_action_for_task(action_text)
            new_hist = (hist or tuple()) + (self.Text(parsed_action, True),)
            appended_histories.append(new_hist)

        # Step the underlying env
        step_results = self.env.step(appended_histories, done=self._dones)

        # Postprocess
        next_histories: List[Optional[Tuple[self.Text, ...]]] = []
        rewards: List[float] = []
        dones: List[bool] = []
        responses: List[str] = []
        for idx, res in enumerate(step_results):
            if res is None:
                # Done slot remains unchanged; LMRL returns None
                next_histories.append(self._histories[idx])
                rewards.append(0.0)
                dones.append(True)
                responses.append("")
                continue
            h, r, d = res
            next_histories.append(h)
            rewards.append(float(r))
            dones.append(bool(d))
            responses.append(self._last_observation_text(h))

        # Update internal state
        self._histories = next_histories
        self._dones = dones

        next_state = [self._history_to_strings(h) for h in self._histories]
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        done_tensor = torch.tensor(dones, dtype=torch.bool)

        return next_state, responses, reward_tensor, done_tensor

    def score_safety(self, responses: List[str], **kwargs) -> torch.Tensor:
        # For general LMRL tasks there is no safety notion by default.
        # Return zeros; callers that rely on safety scoring should disable it in config.
        return torch.zeros(len(responses), dtype=torch.float32)

    # Utilities expected by some trainer code paths
    def get_target_tokenizer(self):
        # Not applicable for LMRL tasks; return None and let caller fall back
        return None

    def get_perplexity_model(self):
        return None

    def get_perplexity_tokenizer(self):
        return None

    def get_target_responses_from_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer: Any) -> List[str]:
        # General envs do not have a target model; return empty responses matching batch size
        batch_size = int(input_ids.size(0))
        return [""] * batch_size

    def score_safety_from_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer: Any) -> torch.Tensor:
        # No safety scoring in general envs; return zeros
        batch_size = int(input_ids.size(0))
        return torch.zeros(batch_size, dtype=torch.float32, device=input_ids.device)

    def _get_embed_similarity_batched(self, sequences1: List[str], sequences2: List[str]) -> torch.Tensor:
        # No embedding similarity in general envs by default; return zeros of len(sequences1)
        if len(sequences1) != len(sequences2):
            raise ValueError("_get_embed_similarity_batched requires equal-sized inputs")
        return torch.zeros(len(sequences1), dtype=torch.float32)

    # --- Helpers ---
    def _history_to_strings(self, history: Optional[Tuple[Any, ...]]) -> List[str]:
        if history is None:
            return []
        return [item.text for item in history]

    def _last_observation_text(self, history: Tuple[Any, ...]) -> str:
        # Take last non-action text if exists, else empty
        for item in reversed(history):
            if not getattr(item, "is_action", False):
                return str(item.text)
        return ""

    def _parse_action_for_task(self, text: str) -> str:
        t = self.task.lower()
        if t == "maze":
            return self._parse_action_maze(text)
        if t == "wordle":
            return self._parse_action_wordle(text)
        if t == "chess":
            return self._parse_action_chess(text)
        # Default: pass-through
        return text

    def _parse_action_maze(self, text: str) -> str:
        lowered = text.lower()
        if "left" in lowered:
            return "move left\n"
        if "right" in lowered:
            return "move right\n"
        if "up" in lowered:
            return "move up\n"
        if "down" in lowered:
            return "move down\n"
        # Also accept exact canonical forms (with/without trailing newline)
        canonical = {"move left", "move right", "move up", "move down"}
        stripped = lowered.strip()
        if stripped in canonical:
            return stripped + "\n"
        raise ValueError(f"Maze action not recognized from agent output: {text!r}")

    def _parse_action_wordle(self, text: str) -> str:
        # Find first 5-letter alphabetical token
        import re
        match = re.search(r"\b([a-zA-Z]{5})\b", text)
        if not match:
            raise ValueError(f"Wordle action requires a 5-letter word, got: {text!r}")
        return match.group(1).lower()

    def _parse_action_chess(self, text: str) -> str:
        # Expect SAN; validate with python-chess if available
        try:
            import chess
        except Exception:
            # If chess is not installed, fail-fast as per code policy
            raise ImportError("python-chess is required for chess task. Install with: pip install chess")
        # We cannot fully validate SAN without a board state here; pass-through and let env validate.
        # Minimal sanity check: no whitespace and reasonable length.
        candidate = text.strip()
        if not candidate:
            raise ValueError("Empty chess action")
        if len(candidate) > 12:
            raise ValueError(f"Unreasonably long chess SAN: {candidate!r}")
        return candidate

    def _parse_action_twenty_questions(self, text: str) -> str:
        # Ensure a well-formed question ending with '?', minimal length checks
        question = (text or "").strip()
        if len(question) == 0:
            return "?\n"
        if question[-1] != "?":
            question += "?"
        # Capitalize first character for cleaner formatting
        if question:
            question = question[0].upper() + question[1:]
        return question + "\n" 