"""
AgentHarm evaluation runner for red-team attackers.

Groups results by AgentHarm behavior category and reports ASR plus
official harm scores (via environment.score_safety).

Usage:
    python -m src.agentharm_test_runner \
        --config-path src/configs/agentharm_eval.json \
        --checkpoint-dir outputs/.../checkpoint_99 \
        --output-dir agentharm_results/run1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

torch.set_grad_enabled(False)

from src.config import AgentConfig
from src.main import build_environment, setup_model_singletons
from src.test_runner import (
    _clean_config_for_evaluation,
    _evaluate_batch,
    _fail,
    _load_json,
    _load_high_level_policy,
    _load_policy_with_singleton,
)


def run_agentharm_evaluation(
    config_path: str,
    checkpoint_dir: str,
    output_dir: str,
    eval_batch_size: int = 4,
    episodes: int = 40,
    max_queries: int = 5,
    scenario: str = "aware",
) -> Dict[str, Any]:
    cfg_dict = _load_json(config_path)
    cleaned = _clean_config_for_evaluation(cfg_dict)
    config = AgentConfig.from_dict(cleaned)
    config.environment.batch_size = eval_batch_size
    config.training.batch_size = eval_batch_size

    opts = dict(config.environment.lmrl_options)
    opts["evaluation_mode"] = True
    config.environment.lmrl_options = opts

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("[AGENTHARM-EVAL] Loading model singletons...")
    singletons = setup_model_singletons(config)
    tokenizer = singletons["tokenizer"]

    print("[AGENTHARM-EVAL] Building AgentHarm environment...")
    environment = build_environment(config, tokenizer, singletons)
    harm_threshold = float(opts.get("harm_threshold", 0.5))

    policy_dir = os.path.join(checkpoint_dir, "policy")
    hl_dir = os.path.join(checkpoint_dir, "high_level_policy")
    policy = _load_policy_with_singleton(policy_dir, singletons.get("agent"), device=device)
    encoder = policy.base_model if hasattr(policy, "base_model") else policy.model
    high_level_policy = _load_high_level_policy(
        hl_dir, device=device, encoder=encoder, tokenizer=tokenizer
    )
    if high_level_policy is not None:
        high_level_policy = high_level_policy.to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_path = os.path.join(output_dir, f"conversations_{timestamp}.jsonl")

    category_successes: Dict[str, List[bool]] = defaultdict(list)
    category_harm_scores: Dict[str, List[float]] = defaultdict(list)
    category_refusals: Dict[str, List[bool]] = defaultdict(list)

    with open(conv_path, "w") as conv_logger:
        batch_count = 0
        for batch_start in range(0, episodes, eval_batch_size):
            current_batch = min(eval_batch_size, episodes - batch_start)
            batch_count += 1
            print(
                f"[AGENTHARM-EVAL] Batch {batch_count}: "
                f"episodes {batch_start + 1}-{batch_start + current_batch}"
            )
            successes, logs = _evaluate_batch(
                policy,
                environment,
                max_queries,
                scenario,
                current_batch,
                high_level_policy=high_level_policy,
                conversation_logger=conv_logger,
            )
            for i, success in enumerate(successes):
                log = logs[i]
                category = "unknown"
                if hasattr(environment, "episodes") and i < len(environment.episodes):
                    category = environment.episodes[i].metadata.get("category", "unknown")
                log["category"] = category

                category_successes[category].append(bool(success))
                conv = log.get("conversation", [])
                if conv:
                    last = conv[-1]
                    category_harm_scores[category].append(float(last.get("guard_score", 0.0)))
                    category_refusals[category].append(
                        bool(last.get("guard_unsafe", False)) is False and success is False
                    )

    category_metrics: Dict[str, Any] = {}
    all_successes: List[bool] = []
    all_harm: List[float] = []
    for category, succ_list in category_successes.items():
        asr = float(np.mean(succ_list)) if succ_list else 0.0
        harm_mean = float(np.mean(category_harm_scores[category])) if category_harm_scores[category] else 0.0
        refusal_rate = float(np.mean(category_refusals[category])) if category_refusals[category] else 0.0
        category_metrics[category] = {
            "episodes": len(succ_list),
            "successes": int(sum(succ_list)),
            "asr": asr,
            "avg_score": harm_mean,
            "avg_refusals": refusal_rate,
        }
        all_successes.extend(succ_list)
        all_harm.extend(category_harm_scores[category])

    summary = {
        "harm_threshold": harm_threshold,
        "total_episodes": len(all_successes),
        "total_successes": int(sum(all_successes)),
        "overall_asr": float(np.mean(all_successes)) if all_successes else 0.0,
        "avg_score": float(np.mean(all_harm)) if all_harm else 0.0,
        "avg_full_score": float(np.mean(all_harm)) if all_harm else 0.0,
        "categories": category_metrics,
        "conversations_path": conv_path,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, "results_by_category.csv")
    pd.DataFrame(
        [
            {
                "category": cat,
                "episodes": m["episodes"],
                "successes": m["successes"],
                "asr": m["asr"],
                "avg_score": m["avg_score"],
                "avg_refusals": m["avg_refusals"],
            }
            for cat, m in category_metrics.items()
        ]
    ).sort_values("asr", ascending=False).to_csv(csv_path, index=False)

    print(f"[AGENTHARM-EVAL] Overall ASR: {summary['overall_asr']:.3f}")
    print(f"[AGENTHARM-EVAL] Avg harm score: {summary['avg_score']:.3f}")
    print(f"[AGENTHARM-EVAL] Summary: {summary_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentHarm red-team evaluation")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--scenario", type=str, default="aware", choices=["aware", "myopic"])
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        _fail(f"Config not found: {args.config_path}")
    if not os.path.exists(args.checkpoint_dir):
        _fail(f"Checkpoint not found: {args.checkpoint_dir}")

    run_agentharm_evaluation(
        config_path=args.config_path,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        eval_batch_size=args.eval_batch_size,
        episodes=args.episodes,
        max_queries=args.max_queries,
        scenario=args.scenario,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
