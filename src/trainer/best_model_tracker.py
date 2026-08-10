"""Running-window best-model checkpoint score tracker."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Deque, Dict, List, Optional


class BestModelTracker:
    """Track rolling window of episode scores and whether to overwrite best_model."""

    def __init__(
        self,
        window: int = 5,
        reward_alpha: float = 0.5,
        tracker_path: Optional[str] = None,
    ):
        if window < 1:
            raise ValueError(f"best_model_window must be >= 1, got {window}")
        self.window = window
        self.reward_alpha = reward_alpha
        self.tracker_path = tracker_path
        self.scores: Deque[float] = deque(maxlen=window)
        self.best_window_score: float = float("-inf")
        self.best_episode: int = -1
        if tracker_path is not None and os.path.isfile(tracker_path):
            self.load(tracker_path)

    def episode_score(self, mean_reward: float, guard_target_mean: Optional[float]) -> float:
        """Composite score: alpha * reward + (1-alpha) * guard_target (if available)."""
        if guard_target_mean is None:
            return float(mean_reward)
        a = self.reward_alpha
        return float(a * mean_reward + (1.0 - a) * guard_target_mean)

    def update(
        self,
        mean_reward: float,
        episode: int,
        guard_target_mean: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Append episode score; return info including whether this window is a new best.

        is_new_best is True only when the window is full and its mean exceeds prior best.
        """
        score = self.episode_score(mean_reward, guard_target_mean)
        self.scores.append(score)
        window_mean = sum(self.scores) / len(self.scores)
        is_new_best = False
        if len(self.scores) >= self.window and window_mean > self.best_window_score:
            self.best_window_score = window_mean
            self.best_episode = episode
            is_new_best = True
        result = {
            "episode_score": score,
            "window_mean": window_mean,
            "window_size": float(len(self.scores)),
            "best_window_score": self.best_window_score,
            "is_new_best": float(1.0 if is_new_best else 0.0),
        }
        if self.tracker_path is not None:
            self.save(self.tracker_path)
        return result

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "window": self.window,
            "reward_alpha": self.reward_alpha,
            "scores": list(self.scores),
            "best_window_score": self.best_window_score,
            "best_episode": self.best_episode,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)
        self.window = int(payload["window"])
        self.reward_alpha = float(payload["reward_alpha"])
        self.scores = deque(payload["scores"], maxlen=self.window)
        self.best_window_score = float(payload["best_window_score"])
        self.best_episode = int(payload["best_episode"])
