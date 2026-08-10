"""Running reward Z-score statistics (EMA)."""

from __future__ import annotations

from typing import Optional

import torch


class RunningRewardStats:
    """EMA mean/variance for reward normalization."""

    def __init__(self, momentum: float = 0.99, clip: float = 10.0, eps: float = 1e-8):
        self.momentum = momentum
        self.clip = clip
        self.eps = eps
        self.mean: Optional[float] = None
        self.var: Optional[float] = None
        self.count: int = 0

    def update(self, x: torch.Tensor) -> None:
        """Update running stats from a detached batch tensor."""
        flat = x.detach().float().reshape(-1)
        if flat.numel() == 0:
            raise ValueError("RunningRewardStats.update called with empty tensor")
        batch_mean = flat.mean().item()
        batch_var = flat.var(unbiased=False).item() if flat.numel() > 1 else 0.0
        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
        else:
            m = self.momentum
            self.mean = m * self.mean + (1.0 - m) * batch_mean
            self.var = m * self.var + (1.0 - m) * batch_var
        self.count += 1

    def normalize(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Z-score normalize, optionally updating stats first."""
        if update:
            self.update(x)
        if self.mean is None or self.var is None:
            raise RuntimeError("RunningRewardStats.normalize called before any update")
        std = (self.var + self.eps) ** 0.5
        z = (x.float() - self.mean) / (std + self.eps)
        if self.clip is not None and self.clip > 0:
            z = torch.clamp(z, min=-self.clip, max=self.clip)
        return z.to(dtype=x.dtype)

    def state_dict(self) -> dict:
        return {
            "momentum": self.momentum,
            "clip": self.clip,
            "eps": self.eps,
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.momentum = state["momentum"]
        self.clip = state["clip"]
        self.eps = state["eps"]
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
