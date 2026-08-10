"""
Utility helpers for trainer logging and checkpoint management.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
from typing import Dict, Optional, Any

import torch


def _safe_template_count(templates_path: str) -> int:
    try:
        with open(templates_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "templates" in data:
            templates = data["templates"]
        else:
            templates = data
        return max(len(templates) - 1, -1)
    except Exception:
        return -1


def log_episode_outputs(
    *,
    run_log_dir: str,
    run_id: str,
    episode_index: int,
    mean_reward: float,
    batch_variance: float,
    batch_std: float,
    templates_path: str,
    metrics: Dict[str, float],
    tokenizer,
    batch_base_state: Optional[torch.Tensor],
    high_level_actions: Optional[torch.Tensor],
) -> None:
    os.makedirs(run_log_dir, exist_ok=True)
    num_templates = _safe_template_count(templates_path)
    experiment_id = f"{run_id}_template_{num_templates}"
    episode_num = episode_index + 1

    # Reward statistics
    rewards_path = os.path.join(run_log_dir, "episodic_rewards.csv")
    rewards_exists = os.path.isfile(rewards_path)
    with open(rewards_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not rewards_exists:
            writer.writerow(
                [
                    "episode",
                    "mean_reward",
                    "batch_variance",
                    "batch_std",
                    "num_templates",
                    "experiment_id",
                    "timestamp",
                ]
            )
        writer.writerow(
            [
                episode_num,
                mean_reward,
                batch_variance,
                batch_std,
                num_templates,
                experiment_id,
                datetime.datetime.now().isoformat(),
            ]
        )

    # All metrics per episode (losses, guard scores, freeform stats, etc.)
    losses_path = os.path.join(run_log_dir, "losses_per_episode.csv")
    # Get all metric keys, sorted for consistent column ordering
    all_metric_keys = sorted([k for k in metrics.keys() if k not in ["experiment_id", "episode"]])
    losses_exists = os.path.isfile(losses_path)
    with open(losses_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode"] + all_metric_keys + ["experiment_id"])
        if not losses_exists:
            writer.writeheader()
        row = {"episode": episode_num, "experiment_id": experiment_id}
        for key in all_metric_keys:
            value = metrics.get(key)
            # Convert to float, handling None and non-numeric types
            if value is None:
                row[key] = float("nan")
            elif isinstance(value, (int, float, bool)):
                row[key] = float(value)
            elif isinstance(value, torch.Tensor):
                row[key] = float(value.item()) if value.numel() == 1 else float("nan")
            else:
                # Skip non-numeric values
                row[key] = float("nan")
        writer.writerow(row)

    # High-level action logging
    if tokenizer is None or batch_base_state is None:
        return

    base_states = batch_base_state.cpu()
    input_texts = [
        tokenizer.decode(tensor.tolist(), skip_special_tokens=True) for tensor in base_states
    ]

    if high_level_actions is not None:
        actions = high_level_actions.detach().cpu().tolist()
    else:
        actions = [-1] * len(input_texts)

    pairs_path = os.path.join(run_log_dir, "input_action_pairs.csv")
    pairs_exists = os.path.isfile(pairs_path)
    with open(pairs_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not pairs_exists:
            writer.writerow(["episode", "input", "action_number", "experiment_id"])
        for text, action_number in zip(input_texts, actions):
            writer.writerow([episode_num, text, action_number, experiment_id])


def save_checkpoint(
    *,
    policy,
    sequence_critic,
    token_critic,
    episode: int,
    metrics: Dict[str, float],
    output_dir: str,
    is_final: bool = False,
    is_best: bool = False,
    high_level_policy=None,
    optimizers: Optional[Dict[str, Optional[torch.optim.Optimizer]]] = None,
    **kwargs,
) -> None:
    if is_best:
        checkpoint_dir = os.path.join(output_dir, "best_model")
    elif is_final:
        checkpoint_dir = os.path.join(output_dir, "final_model")
    else:
        checkpoint_dir = os.path.join(output_dir, f"checkpoint_{episode}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if policy is not None:
        policy.save(os.path.join(checkpoint_dir, "policy"))
    if sequence_critic is not None:
        sequence_critic.save(os.path.join(checkpoint_dir, "sequence_critic"))
    if token_critic is not None:
        token_critic.save(os.path.join(checkpoint_dir, "token_critic"))
    if high_level_policy is not None:
        high_level_policy.save(os.path.join(checkpoint_dir, "high_level_policy"))

    with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if optimizers:
        optim_dir = os.path.join(checkpoint_dir, "optimizer_states")
        os.makedirs(optim_dir, exist_ok=True)
        try:
            actor_opt = optimizers.get("actor")
            if actor_opt is not None:
                torch.save(actor_opt.state_dict(), os.path.join(optim_dir, "actor_optimizer.pt"))
            seq_opt = optimizers.get("sequence_critic")
            if seq_opt is not None:
                torch.save(
                    seq_opt.state_dict(), os.path.join(optim_dir, "sequence_critic_optimizer.pt")
                )
            token_opt = optimizers.get("token_critic")
            if token_opt is not None:
                torch.save(
                    token_opt.state_dict(), os.path.join(optim_dir, "token_critic_optimizer.pt")
                )
            hl_opt = optimizers.get("high_level")
            if hl_opt is not None:
                torch.save(
                    hl_opt.state_dict(), os.path.join(optim_dir, "high_level_optimizer.pt")
                )
        except Exception as exc:  # pragma: no cover - fail fast in production use
            raise RuntimeError(f"Failed to save optimizer states: {exc}") from exc

