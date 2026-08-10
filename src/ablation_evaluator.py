#!/usr/bin/env python3
"""
Ablation Evaluator for Hierarchical vs Flat Agent Comparison

This script evaluates three model configurations:
1. Hierarchical model (with high-level policy) - from checkpoint
2. Flat model (low-level only, trained) - from checkpoint  
3. Non-finetuned baseline (same base model, no training) - no checkpoint

All three models use the same system prompts for fair comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer

from src.config import AgentConfig
from src.base import BaseModelSingleton
from src.environment import MultiStepEnv
from src.policies.low_level import LowLevelPolicy
from src.main import build_environment, build_policy, build_high_level_policy
from src.test_runner import _load_json, _clean_config_for_evaluation, _evaluate_episode, _evaluate_batch, _fail


def evaluate_baseline_model(
    config: AgentConfig,
    model_singletons: Dict[str, Any],
    environment: MultiStepEnv,
    eval_batch_size: int = 8,
    episodes: int = 100,
    max_queries: int = 30
) -> Dict[str, Any]:
    """
    Evaluate a non-finetuned baseline model (no checkpoint loading).
    
    Args:
        config: Agent configuration
        model_singletons: Model singletons dictionary
        environment: Environment instance
        eval_batch_size: Batch size for evaluation
        episodes: Number of episodes to evaluate
        max_queries: Maximum queries per episode
        
    Returns:
        Dictionary with evaluation results
    """
    print("[ABLATION-EVAL] Evaluating non-finetuned baseline model...")
    
    # Create agent singleton for baseline to reuse base model (avoids loading duplicate copy)
    # This is critical: without a singleton, build_policy would load a new full model copy
    if "agent" not in model_singletons:
        print("[ABLATION-EVAL] Creating agent singleton for baseline evaluation...")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        agent_prefer_gpu = 0 if num_gpus > 0 else None
        agent_singleton = BaseModelSingleton.get_instance(
            model_name=config.model.hf_checkpoint,
            use_offloading=config.training.use_offloading,
            config=config,
            prefer_gpu=agent_prefer_gpu,
            quantize=None,  # No quantization for baseline
            instance_tag="agent-baseline",
        )
        model_singletons["agent"] = agent_singleton
    
    # For baseline, ensure we use base model without LoRA adapters
    original_use_lora = config.model.use_lora
    config.model.use_lora = False  # Use base model directly for baseline
    
    # Build policy from base model (no checkpoint) - will use singleton base model
    policy = build_policy(config, model_singletons, environment)
    policy.eval()
    
    # Restore original LoRA setting
    config.model.use_lora = original_use_lora
    
    # No high-level policy for baseline
    high_level_policy = None
    
    # Move to device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    
    # Set tokenizer
    policy.tokenizer = model_singletons.get('tokenizer')
    
    # Attach environment to policy (required for some policy methods)
    policy.env = environment
    
    # Evaluate
    results = _run_evaluation(
        policy=policy,
        environment=environment,
        high_level_policy=high_level_policy,
        config=config,
        eval_batch_size=eval_batch_size,
        episodes=episodes,
        max_queries=max_queries,
        model_name="baseline"
    )
    
    # Free GPU memory after baseline evaluation: move model off device and let it fall out of scope
    print("[ABLATION-EVAL] Cleaning up baseline model from GPU...")
    try:
        # Move policy model to CPU
        if hasattr(policy, 'model') and policy.model is not None:
            policy.model.to('cpu')
        
        # Clear reference and let object fall out of scope
        del policy
        
        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()  # Second pass to catch any circular references
        torch.cuda.empty_cache()
        print("[ABLATION-EVAL] Cleanup complete for baseline")
    except Exception as e:
        print(f"[ABLATION-EVAL] Warning: Error during baseline cleanup: {e}")
    
    return results


def evaluate_from_checkpoint(
    config: AgentConfig,
    checkpoint_dir: str,
    model_singletons: Dict[str, Any],
    environment: MultiStepEnv,
    eval_batch_size: int = 8,
    episodes: int = 100,
    max_queries: int = 30,
    model_name: str = "checkpoint"
) -> Dict[str, Any]:
    """
    Evaluate a model from checkpoint.
    
    Args:
        config: Agent configuration
        checkpoint_dir: Directory containing checkpoint
        model_singletons: Model singletons dictionary
        environment: Environment instance
        eval_batch_size: Batch size for evaluation
        episodes: Number of episodes to evaluate
        max_queries: Maximum queries per episode
        model_name: Name for logging
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"[ABLATION-EVAL] Evaluating {model_name} model from checkpoint: {checkpoint_dir}")
    
    from src.main import load_models_from_checkpoint
    
    # Load models from checkpoint
    loaded_models = load_models_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        config=config,
        model_singletons=model_singletons,
        force=False
    )
    
    policy = loaded_models.get("policy")
    if policy is None:
        _fail(f"Policy not found in checkpoint: {checkpoint_dir}")
    
    policy.eval()
    
    high_level_policy = loaded_models.get("high_level_policy")
    if high_level_policy is not None:
        high_level_policy.eval()
    
    # Move to device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    if high_level_policy is not None:
        high_level_policy.to(device)
    
    # Set tokenizers
    policy.tokenizer = model_singletons.get('tokenizer')
    # Freeform HL keeps the tokenizer loaded with its PEFT base model.
    if high_level_policy is not None and getattr(high_level_policy, "policy_kind", None) != "freeform":
        high_level_policy.tokenizer = model_singletons.get('tokenizer')
    
    # Attach environment to policy (required for some policy methods)
    policy.env = environment
    
    # Evaluate
    results = _run_evaluation(
        policy=policy,
        environment=environment,
        high_level_policy=high_level_policy,
        config=config,
        eval_batch_size=eval_batch_size,
        episodes=episodes,
        max_queries=max_queries,
        model_name=model_name
    )

    # Free GPU memory after evaluation: move models off device and let them fall out of scope
    print(f"[ABLATION-EVAL] Cleaning up {model_name} model from GPU...")
    try:
        # Move policy model to CPU
        if hasattr(policy, 'model') and policy.model is not None:
            policy.model.to('cpu')
        # Move high-level policy model to CPU if it exists
        if high_level_policy is not None:
            if hasattr(high_level_policy, 'model') and high_level_policy.model is not None:
                high_level_policy.model.to('cpu')
            elif hasattr(high_level_policy, 'base_model') and high_level_policy.base_model is not None:
                high_level_policy.base_model.to('cpu')
        
        # Clear references and let objects fall out of scope
        del loaded_models
        del policy
        if high_level_policy is not None:
            del high_level_policy
        
        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()  # Second pass to catch any circular references
        torch.cuda.empty_cache()
        print(f"[ABLATION-EVAL] Cleanup complete for {model_name}")
    except Exception as e:
        print(f"[ABLATION-EVAL] Warning: Error during cleanup: {e}")

    return results


def _run_evaluation(
    policy: Any,
    environment: MultiStepEnv,
    high_level_policy: Optional[Any],
    config: AgentConfig,
    eval_batch_size: int,
    episodes: int,
    max_queries: int,
    model_name: str
) -> Dict[str, Any]:
    """
    Run evaluation episodes and collect results.
    
    Args:
        policy: Policy to evaluate
        environment: Environment instance
        high_level_policy: Optional high-level policy
        config: Agent configuration
        eval_batch_size: Batch size for evaluation
        episodes: Number of episodes
        max_queries: Maximum queries per episode
        model_name: Name for logging
        
    Returns:
        Dictionary with evaluation results including diversity and TTS metrics
    """
    print(f"[ABLATION-EVAL] Running {episodes} evaluation episodes for {model_name}...")
    
    templates_path = getattr(config.model, 'templates_path', 'src/configs/templates.json')
    
    all_successes = []
    all_conversation_logs = []
    
    # Run evaluation in batches
    num_batches = (episodes + eval_batch_size - 1) // eval_batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * eval_batch_size
        batch_end = min(batch_start + eval_batch_size, episodes)
        actual_batch_size = batch_end - batch_start
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"[ABLATION-EVAL] {model_name}: Batch {batch_idx + 1}/{num_batches} (episodes {batch_start+1}-{batch_end})")
        
        batch_successes, batch_logs = _evaluate_batch(
            policy=policy,
            environment=environment,
            max_queries=max_queries,
            scenario="aware",  # Use aware target (full conversation history)
            batch_size=actual_batch_size,
            high_level_policy=high_level_policy,
            templates_path=templates_path
        )
        
        all_successes.extend(batch_successes)
        all_conversation_logs.extend(batch_logs)
    
    # Compute basic metrics
    success_rate = sum(all_successes) / len(all_successes) if all_successes else 0.0
    
    # Compute TTS (Time To Success) metrics
    queries_to_success = []
    for log in all_conversation_logs:
        if log.get('success', False):
            queries_used = log.get('queries_used', 0)
            if queries_used > 0:
                queries_to_success.append(queries_used)
    
    queries_to_success_mean = float(np.mean(queries_to_success)) if queries_to_success else 0.0
    queries_to_success_std = float(np.std(queries_to_success)) if len(queries_to_success) > 1 else 0.0
    queries_to_success_min = float(np.min(queries_to_success)) if queries_to_success else 0.0
    queries_to_success_max = float(np.max(queries_to_success)) if queries_to_success else 0.0
    first_turn_success_rate = sum(1 for q in queries_to_success if q == 1) / len(queries_to_success) if queries_to_success else 0.0
    
    # Compute diversity metrics from actions
    all_actions = []
    for log in all_conversation_logs:
        conversation = log.get('conversation', [])
        for turn in conversation:
            agent_message = turn.get('agent_message', '')
            if agent_message:
                all_actions.append(agent_message.strip())
    
    unique_actions = set(all_actions)
    total_actions = len(all_actions)
    unique_action_count = len(unique_actions)
    diversity_ratio = unique_action_count / total_actions if total_actions > 0 else 0.0
    
    # Compute agent guard score metrics (detectability of agent actions)
    agent_guard_scores = []
    for log in all_conversation_logs:
        conversation = log.get('conversation', [])
        for turn in conversation:
            agent_guard_score = turn.get('agent_guard_score')
            if agent_guard_score is not None:
                agent_guard_scores.append(float(agent_guard_score))
    
    agent_guard_mean = float(np.mean(agent_guard_scores)) if agent_guard_scores else 0.0
    agent_guard_std = float(np.std(agent_guard_scores)) if len(agent_guard_scores) > 1 else 0.0
    agent_guard_min = float(np.min(agent_guard_scores)) if agent_guard_scores else 0.0
    agent_guard_max = float(np.max(agent_guard_scores)) if agent_guard_scores else 0.0
    agent_guard_high_detectability_rate = sum(1 for s in agent_guard_scores if s > 0.5) / len(agent_guard_scores) if agent_guard_scores else 0.0
    
    results = {
        "model_name": model_name,
        "episodes": episodes,
        "max_queries": max_queries,
        "success_rate": success_rate,
        "successes": sum(all_successes),
        "total": len(all_successes),
        "success_list": all_successes,
        # TTS metrics
        "queries_to_success_mean": queries_to_success_mean,
        "queries_to_success_std": queries_to_success_std,
        "queries_to_success_min": queries_to_success_min,
        "queries_to_success_max": queries_to_success_max,
        "first_turn_success_rate": first_turn_success_rate,
        # Diversity metrics
        "unique_action_count": unique_action_count,
        "total_action_count": total_actions,
        "diversity_ratio": diversity_ratio,
        # Agent guard score metrics (detectability)
        "agent_guard_mean": agent_guard_mean,
        "agent_guard_std": agent_guard_std,
        "agent_guard_min": agent_guard_min,
        "agent_guard_max": agent_guard_max,
        "agent_guard_high_detectability_rate": agent_guard_high_detectability_rate,  # % of actions with score > 0.5
        "agent_guard_score_count": len(agent_guard_scores)
    }
    
    print(f"[ABLATION-EVAL] {model_name} Results:")
    print(f"  Success Rate: {success_rate:.2%} ({sum(all_successes)}/{len(all_successes)})")
    print(f"  TTS (queries to success): {queries_to_success_mean:.2f} ± {queries_to_success_std:.2f} (range: [{queries_to_success_min:.0f}, {queries_to_success_max:.0f}])")
    print(f"  First turn success rate: {first_turn_success_rate:.2%}")
    print(f"  Diversity: {unique_action_count} unique actions out of {total_actions} total (ratio: {diversity_ratio:.3f})")
    print(f"  Agent Guard Score (detectability): {agent_guard_mean:.4f} ± {agent_guard_std:.4f} (range: [{agent_guard_min:.4f}, {agent_guard_max:.4f}])")
    print(f"  High detectability rate (score > 0.5): {agent_guard_high_detectability_rate:.2%}")
    
    return results


def run_ablation_evaluation(
    config_path: str,
    hierarchical_checkpoint: Optional[str] = None,
    hierarchical_checkpoints: Optional[List[str]] = None,
    flat_checkpoint: Optional[str] = None,
    baseline_model_name: Optional[str] = None,
    eval_batch_size: int = 8,
    episodes: int = 100,
    max_queries: int = 30,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run ablation evaluation comparing hierarchical, flat, and baseline models.
    
    Args:
        config_path: Path to configuration file
        hierarchical_checkpoint: Path to hierarchical model checkpoint (optional)
        flat_checkpoint: Path to flat model checkpoint (optional)
        baseline_model_name: Base model name for baseline (optional, uses config if not provided)
        eval_batch_size: Batch size for evaluation
        episodes: Number of episodes per model
        max_queries: Maximum queries per episode
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all evaluation results
    """
    print("[ABLATION-EVAL] Starting ablation evaluation...")
    
    # Load configuration
    cfg_dict = _load_json(config_path)
    cleaned_cfg_dict = _clean_config_for_evaluation(cfg_dict)
    config = AgentConfig.from_dict(cleaned_cfg_dict)
    
    # Lightweight setup for evaluation:
    # - Do NOT instantiate agent/critic BaseModelSingletons (saves GPU 0 memory)
    # - Only build tokenizer + target/guard models for the environment.
    print("[ABLATION-EVAL] Setting up environment models and tokenizer (evaluation mode)...")

    from src.utils import ensure_tokenizer_padding_settings

    # Main tokenizer for the agent policy
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.hf_checkpoint,
        use_fast=True,
        padding_side="left",
    )
    tokenizer.padding_side = "left"
    tokenizer = ensure_tokenizer_padding_settings(tokenizer)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Target model (inference GPU, typically GPU 1)
    target_singleton = None
    target_tokenizer = None
    if getattr(config.environment, "target_model", None):
        target_model_name = config.environment.target_model
        openai_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-5"]
        is_openai_target = any(str(target_model_name).startswith(model) for model in openai_models)
        if is_openai_target:
            print(f"OpenAI target model detected: {target_model_name} (handled by environment)")
        else:
            target_prefer_gpu = 1 if num_gpus > 1 else None
            print(f"Loading target model singleton for evaluation: {target_model_name}")
            if target_prefer_gpu is not None:
                print(f"  -> placing TARGET model on GPU {target_prefer_gpu}")
            target_quantize = getattr(config.environment, "target_quantize", None)
            print(f"Target model quantization setting (eval): {target_quantize}")
            target_singleton = BaseModelSingleton.get_instance(
                model_name=target_model_name,
                use_offloading=config.training.use_offloading,
                config=config,
                prefer_gpu=target_prefer_gpu,
                quantize=target_quantize,
                instance_tag="target-eval",
            )
            target_tokenizer = AutoTokenizer.from_pretrained(
                target_model_name,
                use_fast=True,
                padding_side="left",
            )
            target_tokenizer.padding_side = "left"
            target_tokenizer = ensure_tokenizer_padding_settings(target_tokenizer)
            target_singleton.tokenizer = target_tokenizer

    # Guard model (also on inference GPU when present)
    guard_singleton = None
    guard_tokenizer = None
    guard_model_name = getattr(config.environment, "guard_model", None)
    if guard_model_name:
        guard_prefer_gpu = 1 if num_gpus > 1 else None
        print(f"Loading guard model singleton for evaluation: {guard_model_name}")
        if guard_prefer_gpu is not None:
            print(f"  -> placing GUARD model on GPU {guard_prefer_gpu}")
        guard_quantize = getattr(config.environment, "guard_quantize", None)
        print(f"Guard model quantization setting (eval): {guard_quantize}")
        guard_singleton = BaseModelSingleton.get_instance(
            model_name=guard_model_name,
            use_offloading=config.training.use_offloading,
            config=config,
            prefer_gpu=guard_prefer_gpu,
            quantize=guard_quantize,
            instance_tag="guard-eval",
        )
        guard_tokenizer = AutoTokenizer.from_pretrained(
            guard_model_name,
            use_fast=True,
            padding_side="left",
        )
        guard_tokenizer.padding_side = "left"
        guard_tokenizer = ensure_tokenizer_padding_settings(guard_tokenizer)
        guard_singleton.tokenizer = guard_tokenizer

    model_singletons: Dict[str, Any] = {
        "tokenizer": tokenizer,
        "target": target_singleton,
        "target_tokenizer": target_tokenizer,
        "guard": guard_singleton,
        "guard_tokenizer": guard_tokenizer,
    }

    # Build environment
    print("[ABLATION-EVAL] Building environment...")
    hf_token = os.environ.get("HF_TOKEN") or None
    environment = MultiStepEnv(
        config=config.environment,
        hf_token=hf_token,
        target_singleton=target_singleton,
        target_tokenizer=target_tokenizer,
        guard_singleton=guard_singleton,
        guard_tokenizer=guard_tokenizer,
        debug_mode=config.training.debug_mode,
        max_new_tokens=config.environment.max_new_tokens,
        target_quantize=getattr(config.environment, "target_quantize", None),
        guard_quantize=getattr(config.environment, "guard_quantize", None),
        evaluation_mode=True,
    )
    
    all_results = {}
    
    # 1. Evaluate hierarchical model(s) (if checkpoint(s) provided)
    checkpoints_to_eval: List[str] = []
    if hierarchical_checkpoints:
        checkpoints_to_eval = [ckpt for ckpt in hierarchical_checkpoints if ckpt and os.path.exists(ckpt)]
        missing = [ckpt for ckpt in hierarchical_checkpoints if ckpt and not os.path.exists(ckpt)]
        for ckpt in missing:
            print(f"[ABLATION-EVAL] Skipping hierarchical checkpoint (not found): {ckpt}")
    elif hierarchical_checkpoint and os.path.exists(hierarchical_checkpoint):
        checkpoints_to_eval = [hierarchical_checkpoint]
    else:
        if hierarchical_checkpoint:
            print(f"[ABLATION-EVAL] Skipping hierarchical checkpoint (not found): {hierarchical_checkpoint}")
    
    if checkpoints_to_eval:
        for idx, ckpt in enumerate(checkpoints_to_eval):
            print("\n" + "="*80)
            print(f"EVALUATING HIERARCHICAL MODEL {idx+1}/{len(checkpoints_to_eval)}")
            print("="*80)
            # Temporarily enable high-level policy for hierarchical model
            original_use_hl = config.model.use_high_level_policy
            config.model.use_high_level_policy = True
            
            ckpt_name = os.path.basename(os.path.normpath(ckpt))
            hierarchical_results = evaluate_from_checkpoint(
                config=config,
                checkpoint_dir=ckpt,
                model_singletons=model_singletons,
                environment=environment,
                eval_batch_size=eval_batch_size,
                episodes=episodes,
                max_queries=max_queries,
                model_name=f"hierarchical_{ckpt_name}"
            )
            all_results[f"hierarchical_{ckpt_name}"] = hierarchical_results
            
            # Restore original config
            config.model.use_high_level_policy = original_use_hl
            
            # Aggressive cleanup between hierarchical model evaluations
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    else:
        print("[ABLATION-EVAL] Skipping hierarchical model(s) (no valid checkpoints provided)")
    
    # 2. Evaluate flat model (if checkpoint provided)
    if flat_checkpoint and os.path.exists(flat_checkpoint):
        print("\n" + "="*80)
        print("EVALUATING FLAT MODEL")
        print("="*80)
        # Ensure high-level policy is disabled for flat model
        original_use_hl = config.model.use_high_level_policy
        config.model.use_high_level_policy = False
        
        flat_results = evaluate_from_checkpoint(
            config=config,
            checkpoint_dir=flat_checkpoint,
            model_singletons=model_singletons,
            environment=environment,
            eval_batch_size=eval_batch_size,
            episodes=episodes,
            max_queries=max_queries,
            model_name="flat"
        )
        all_results["flat"] = flat_results
        
        # Restore original config
        config.model.use_high_level_policy = original_use_hl
        
        # Aggressive cleanup before baseline evaluation
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("[ABLATION-EVAL] Skipping flat model (checkpoint not provided or not found)")
    
    # 3. Evaluate baseline model (non-finetuned)
    print("\n" + "="*80)
    print("EVALUATING BASELINE MODEL (NON-FINETUNED)")
    print("="*80)
    # Use provided model name or from config
    if baseline_model_name:
        config.model.hf_checkpoint = baseline_model_name
    # Ensure high-level policy is disabled for baseline
    original_use_hl = config.model.use_high_level_policy
    config.model.use_high_level_policy = False
    
    baseline_results = evaluate_baseline_model(
        config=config,
        model_singletons=model_singletons,
        environment=environment,
        eval_batch_size=eval_batch_size,
        episodes=episodes,
        max_queries=max_queries
    )
    all_results["baseline"] = baseline_results
    
    # Restore original config
    config.model.use_high_level_policy = original_use_hl
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION EVALUATION SUMMARY")
    print("="*80)
    for model_name, results in all_results.items():
        print(f"{model_name.upper()}: {results['success_rate']:.2%} ({results['successes']}/{results['total']})")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"ablation_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n[ABLATION-EVAL] Results saved to: {results_path}")
    
    return all_results


def run_multi_ablation_evaluation(
    config_paths: List[str],
    hierarchical_checkpoints: List[Optional[str]],
    flat_checkpoint: Optional[str] = None,
    baseline_model_name: Optional[str] = None,
    eval_batch_size: int = 8,
    episodes: int = 100,
    max_queries: int = 30,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run ablation evaluation for multiple checkpoints, evaluating each checkpoint only once.
    
    This function:
      - Accepts a list of config paths and matching hierarchical checkpoints
      - Evaluates each hierarchical checkpoint once (with its corresponding config)
      - Evaluates flat checkpoint once (if provided, uses first config)
      - Evaluates baseline model once (if provided, uses first config)
      - Aggregates results into a single dictionary
    """
    if len(config_paths) != len(hierarchical_checkpoints):
        raise ValueError(
            f"config_paths (len={len(config_paths)}) and hierarchical_checkpoints "
            f"(len={len(hierarchical_checkpoints)}) must have the same length."
        )
    
    base_output_dir = output_dir or "ablation_results"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Use first config for environment setup (all configs should have same env settings)
    primary_config_path = config_paths[0]
    if not os.path.exists(primary_config_path):
        raise FileNotFoundError(f"Config path does not exist: {primary_config_path}")
    
    # Load primary configuration for environment setup
    cfg_dict = _load_json(primary_config_path)
    cleaned_cfg_dict = _clean_config_for_evaluation(cfg_dict)
    config = AgentConfig.from_dict(cleaned_cfg_dict)
    
    # Setup environment and model singletons (shared across all evaluations)
    print("[ABLATION-EVAL] Setting up environment models and tokenizer (evaluation mode)...")
    
    from src.utils import ensure_tokenizer_padding_settings
    
    # Main tokenizer for the agent policy
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.hf_checkpoint,
        use_fast=True,
        padding_side="left",
    )
    tokenizer.padding_side = "left"
    tokenizer = ensure_tokenizer_padding_settings(tokenizer)
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Target model (inference GPU, typically GPU 1)
    target_singleton = None
    target_tokenizer = None
    if getattr(config.environment, "target_model", None):
        target_model_name = config.environment.target_model
        openai_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-5"]
        is_openai_target = any(str(target_model_name).startswith(model) for model in openai_models)
        if is_openai_target:
            print(f"OpenAI target model detected: {target_model_name} (handled by environment)")
        else:
            target_prefer_gpu = 1 if num_gpus > 1 else None
            print(f"Loading target model singleton for evaluation: {target_model_name}")
            if target_prefer_gpu is not None:
                print(f"  -> placing TARGET model on GPU {target_prefer_gpu}")
            target_quantize = getattr(config.environment, "target_quantize", None)
            print(f"Target model quantization setting (eval): {target_quantize}")
            target_singleton = BaseModelSingleton.get_instance(
                model_name=target_model_name,
                use_offloading=config.training.use_offloading,
                config=config,
                prefer_gpu=target_prefer_gpu,
                quantize=target_quantize,
                instance_tag="target-eval",
            )
            target_tokenizer = AutoTokenizer.from_pretrained(
                target_model_name,
                use_fast=True,
                padding_side="left",
            )
            target_tokenizer.padding_side = "left"
            target_tokenizer = ensure_tokenizer_padding_settings(target_tokenizer)
            target_singleton.tokenizer = target_tokenizer
    
    # Guard model (also on inference GPU when present)
    guard_singleton = None
    guard_tokenizer = None
    guard_model_name = getattr(config.environment, "guard_model", None)
    if guard_model_name:
        guard_prefer_gpu = 1 if num_gpus > 1 else None
        print(f"Loading guard model singleton for evaluation: {guard_model_name}")
        if guard_prefer_gpu is not None:
            print(f"  -> placing GUARD model on GPU {guard_prefer_gpu}")
        guard_quantize = getattr(config.environment, "guard_quantize", None)
        print(f"Guard model quantization setting (eval): {guard_quantize}")
        guard_singleton = BaseModelSingleton.get_instance(
            model_name=guard_model_name,
            use_offloading=config.training.use_offloading,
            config=config,
            prefer_gpu=guard_prefer_gpu,
            quantize=guard_quantize,
            instance_tag="guard-eval",
        )
        guard_tokenizer = AutoTokenizer.from_pretrained(
            guard_model_name,
            use_fast=True,
            padding_side="left",
        )
        guard_tokenizer.padding_side = "left"
        guard_tokenizer = ensure_tokenizer_padding_settings(guard_tokenizer)
        guard_singleton.tokenizer = guard_tokenizer
    
    model_singletons: Dict[str, Any] = {
        "tokenizer": tokenizer,
        "target": target_singleton,
        "target_tokenizer": target_tokenizer,
        "guard": guard_singleton,
        "guard_tokenizer": guard_tokenizer,
    }
    
    # Build environment (shared across all evaluations)
    print("[ABLATION-EVAL] Building environment...")
    hf_token = os.environ.get("HF_TOKEN") or None
    environment = MultiStepEnv(
        config=config.environment,
        hf_token=hf_token,
        target_singleton=target_singleton,
        target_tokenizer=target_tokenizer,
        guard_singleton=guard_singleton,
        guard_tokenizer=guard_tokenizer,
        debug_mode=config.training.debug_mode,
        max_new_tokens=config.environment.max_new_tokens,
        target_quantize=getattr(config.environment, "target_quantize", None),
        guard_quantize=getattr(config.environment, "guard_quantize", None),
        evaluation_mode=True,
    )
    
    all_results: Dict[str, Any] = {}
    
    # Track which checkpoints have been evaluated to avoid duplicates
    evaluated_checkpoints: set = set()
    
    # 1. Evaluate hierarchical checkpoints (each with its own config)
    for idx, (cfg_path, hier_ckpt) in enumerate(zip(config_paths, hierarchical_checkpoints)):
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config path does not exist: {cfg_path}")
        
        if hier_ckpt and os.path.exists(hier_ckpt):
            # Skip if already evaluated
            if hier_ckpt in evaluated_checkpoints:
                print(f"[ABLATION-EVAL] Skipping duplicate checkpoint: {hier_ckpt}")
                continue
            
            evaluated_checkpoints.add(hier_ckpt)
            
            print("\n" + "=" * 80)
            print(f"EVALUATING HIERARCHICAL CHECKPOINT {idx+1}/{len(config_paths)}")
            print(f"Config: {cfg_path}")
            print(f"Checkpoint: {hier_ckpt}")
            print("=" * 80)
            
            # Load config for this checkpoint
            ckpt_cfg_dict = _load_json(cfg_path)
            cleaned_ckpt_cfg_dict = _clean_config_for_evaluation(ckpt_cfg_dict)
            ckpt_config = AgentConfig.from_dict(cleaned_ckpt_cfg_dict)
            
            # Temporarily enable high-level policy for hierarchical model
            original_use_hl = ckpt_config.model.use_high_level_policy
            ckpt_config.model.use_high_level_policy = True
            
            ckpt_name = os.path.basename(os.path.normpath(hier_ckpt))
            hierarchical_results = evaluate_from_checkpoint(
                config=ckpt_config,
                checkpoint_dir=hier_ckpt,
                model_singletons=model_singletons,
                environment=environment,
                eval_batch_size=eval_batch_size,
                episodes=episodes,
                max_queries=max_queries,
                model_name=f"hierarchical_{ckpt_name}"
            )
            all_results[f"hierarchical_{ckpt_name}"] = hierarchical_results
            
            # Restore original config
            ckpt_config.model.use_high_level_policy = original_use_hl
            
            # Aggressive cleanup between evaluations
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        elif hier_ckpt:
            print(f"[ABLATION-EVAL] Skipping hierarchical checkpoint (not found): {hier_ckpt}")
    
    # 2. Evaluate flat checkpoint once (if provided and not already evaluated)
    if flat_checkpoint and os.path.exists(flat_checkpoint):
        if flat_checkpoint not in evaluated_checkpoints:
            evaluated_checkpoints.add(flat_checkpoint)
            
            print("\n" + "=" * 80)
            print("EVALUATING FLAT MODEL")
            print("=" * 80)
            
            # Use first config for flat model
            flat_cfg_dict = _load_json(primary_config_path)
            cleaned_flat_cfg_dict = _clean_config_for_evaluation(flat_cfg_dict)
            flat_config = AgentConfig.from_dict(cleaned_flat_cfg_dict)
            
            # Ensure high-level policy is disabled for flat model
            original_use_hl = flat_config.model.use_high_level_policy
            flat_config.model.use_high_level_policy = False
            
            flat_results = evaluate_from_checkpoint(
                config=flat_config,
                checkpoint_dir=flat_checkpoint,
                model_singletons=model_singletons,
                environment=environment,
                eval_batch_size=eval_batch_size,
                episodes=episodes,
                max_queries=max_queries,
                model_name="flat"
            )
            all_results["flat"] = flat_results
            
            # Restore original config
            flat_config.model.use_high_level_policy = original_use_hl
            
            # Aggressive cleanup
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"[ABLATION-EVAL] Skipping flat checkpoint (already evaluated): {flat_checkpoint}")
    elif flat_checkpoint:
        print(f"[ABLATION-EVAL] Skipping flat checkpoint (not found): {flat_checkpoint}")
    
    # 3. Evaluate baseline model once (if provided)
    if baseline_model_name:
        print("\n" + "=" * 80)
        print("EVALUATING BASELINE MODEL (NON-FINETUNED)")
        print("=" * 80)
        
        # Use first config for baseline
        baseline_cfg_dict = _load_json(primary_config_path)
        cleaned_baseline_cfg_dict = _clean_config_for_evaluation(baseline_cfg_dict)
        baseline_config = AgentConfig.from_dict(cleaned_baseline_cfg_dict)
        
        # Use provided model name
        baseline_config.model.hf_checkpoint = baseline_model_name
        # Ensure high-level policy is disabled for baseline
        original_use_hl = baseline_config.model.use_high_level_policy
        baseline_config.model.use_high_level_policy = False
        
        baseline_results = evaluate_baseline_model(
            config=baseline_config,
            model_singletons=model_singletons,
            environment=environment,
            eval_batch_size=eval_batch_size,
            episodes=episodes,
            max_queries=max_queries
        )
        all_results["baseline"] = baseline_results
        
        # Restore original config
        baseline_config.model.use_high_level_policy = original_use_hl
    
    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION EVALUATION SUMMARY")
    print("=" * 80)
    for model_name, results in all_results.items():
        print(f"{model_name.upper()}: {results['success_rate']:.2%} ({results['successes']}/{results['total']})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = os.path.join(base_output_dir, f"multi_ablation_results_{timestamp}.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[ABLATION-EVAL] Combined results saved to: {combined_path}")
    return all_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run ablation evaluation comparing hierarchical, flat, and baseline models"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default=None,
        help="Path to configuration JSON file (required unless --config_paths is used)"
    )
    parser.add_argument(
        "--config_paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of configuration JSON files for multi-study evaluation"
    )
    
    parser.add_argument(
        "--hierarchical_checkpoint",
        type=str,
        default=None,
        help="Path to a single hierarchical model checkpoint directory"
    )
    parser.add_argument(
        "--hierarchical_checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of hierarchical checkpoint directories. If --config_paths is provided, this list must align in length; otherwise all listed checkpoints are evaluated against the single --config_path."
    )
    
    parser.add_argument(
        "--flat_checkpoint",
        type=str,
        default=None,
        help="Path to flat model checkpoint directory"
    )
    
    parser.add_argument(
        "--baseline_model_name",
        type=str,
        default=None,
        help="Base model name for baseline evaluation (uses config if not provided)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per model"
    )
    
    parser.add_argument(
        "--max_queries",
        type=int,
        default=30,
        help="Maximum queries per episode"
    )
    
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    try:
        # Multi-study mode: use lists of config paths and checkpoints
        if args.config_paths is not None:
            if args.hierarchical_checkpoints is None:
                raise ValueError("When using --config_paths, you must also provide --hierarchical_checkpoints.")
            results = run_multi_ablation_evaluation(
                config_paths=args.config_paths,
                hierarchical_checkpoints=args.hierarchical_checkpoints,
                flat_checkpoint=args.flat_checkpoint,
                baseline_model_name=args.baseline_model_name,
                eval_batch_size=args.eval_batch_size,
                episodes=args.episodes,
                max_queries=args.max_queries,
                output_dir=args.output_dir or "ablation_results",
            )
        else:
            # Single-study mode: require config_path
            if args.config_path is None:
                raise ValueError("Either --config_path (single study) or --config_paths (multi study) must be provided.")
            results = run_ablation_evaluation(
                config_path=args.config_path,
                hierarchical_checkpoint=args.hierarchical_checkpoint,
                hierarchical_checkpoints=args.hierarchical_checkpoints,
                flat_checkpoint=args.flat_checkpoint,
                baseline_model_name=args.baseline_model_name,
                eval_batch_size=args.eval_batch_size,
                episodes=args.episodes,
                max_queries=args.max_queries,
                output_dir=args.output_dir or "ablation_results"
            )
        
        print("\n[ABLATION-EVAL] Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"[ABLATION-EVAL] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

