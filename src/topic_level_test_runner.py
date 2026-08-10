"""
Topic-Level Test Runner for trained agents.

This utility extends the standard test runner to evaluate red-teaming agents
across different harm topic categories from the HarmBench dataset. It partitions
the dataset by SemanticCategory and computes topic-specific attack success rates.

Usage:
    python src/topic_level_test_runner.py \
        --config-path configs/topic_level_eval_config.json \
        --checkpoint-dir outputs/normal-agent-8b8t-std/run_20250801_195707/final_model \
        --output-dir topic_level_results/normal-agent-8b8t-std \
        --eval-batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np

import torch

# Disable autograd globally for this module.
# This script is only used for evaluation, so no gradients are needed.
torch.set_grad_enabled(False)

# Import from test_runner
from src.test_runner import (
    _load_json, 
    _clean_config_for_evaluation,
    _fail,
    _load_policy_with_singleton,
    _load_high_level_policy,
    _evaluate_batch
)
from src.config import AgentConfig
from src.environment import MultiStepEnv
from src.base import BaseModelSingleton
from transformers import AutoTokenizer
from src.semantic_analysis.harmbench_loader import (
    load_harmbench_data,
    filter_harmbench_data,
    sample_harmbench_data,
    get_harmbench_statistics
)
from src.diversity_metrics import (
    compute_mean_cosine_distance,
    compute_complexity,
    compute_diversity_and_complexity_per_category
)


# ------------------------------------------------------------------------------
# Topic-Level Evaluation Core
# ------------------------------------------------------------------------------

def load_and_partition_by_topic(harmbench_path: str, category_column: str = "SemanticCategory") -> Dict[str, pd.DataFrame]:
    """
    Load HarmBench dataset and partition by topic categories.
    
    Args:
        harmbench_path: Path to HarmBench CSV file
        category_column: Column name for topic categories
        
    Returns:
        Dictionary mapping topic names to DataFrames
    """
    print(f"[TOPIC-EVAL] Loading HarmBench dataset from: {harmbench_path}")
    
    df = pd.read_csv(harmbench_path)
    print(f"[TOPIC-EVAL] Loaded {len(df)} total entries")
    
    # Check for required columns
    if category_column not in df.columns:
        _fail(f"Category column '{category_column}' not found in dataset. Available: {list(df.columns)}")
    if 'Behavior' not in df.columns:
        _fail(f"'Behavior' column not found in dataset. Available: {list(df.columns)}")
    
    # Get topic distribution
    topic_counts = df[category_column].value_counts()
    print(f"\n[TOPIC-EVAL] Topic distribution:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} ({count/len(df)*100:.1f}%)")
    
    # Partition by topic
    topics = {}
    for topic in topic_counts.index:
        topic_df = df[df[category_column] == topic].copy()
        topics[topic] = topic_df
        print(f"[TOPIC-EVAL] Partitioned '{topic}': {len(topic_df)} entries")
    
    return topics


def create_topic_dataset_file(topic_df: pd.DataFrame, topic_name: str, output_dir: str) -> str:
    """
    Create a temporary CSV file for a specific topic's dataset.
    
    Args:
        topic_df: DataFrame with topic-specific prompts
        topic_name: Name of the topic
        output_dir: Directory to save temporary file
        
    Returns:
        Path to the created CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    topic_file = os.path.join(output_dir, f"topic_{topic_name}.csv")
    topic_df.to_csv(topic_file, index=False)
    
    print(f"[TOPIC-EVAL] Created topic dataset file: {topic_file}")
    return topic_file


def load_models_for_evaluation(
    config: AgentConfig,
    checkpoint_dir: str,
    critic_checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load all models once for reuse across multiple evaluations.
    
    Args:
        config: Agent configuration
        checkpoint_dir: Path to model checkpoint (policy + HL)
        critic_checkpoint_dir: Optional final_model dir whose sequence_critic/
            is used for HL strategy selection (defaults to checkpoint_dir)
        
    Returns:
        Dictionary containing all loaded models and components
    """
    print("\n[TOPIC-EVAL] ========================================")
    print("[TOPIC-EVAL] LOADING MODELS (once for all topics)")
    print("[TOPIC-EVAL] ========================================")
    
    policy_dir = os.path.join(checkpoint_dir, "policy")
    high_level_policy_dir = os.path.join(checkpoint_dir, "high_level_policy")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load model singletons
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    target_prefer_gpu = 1 if num_gpus > 1 else 0 if num_gpus == 1 else None
    guard_prefer_gpu = 1 if num_gpus > 1 else 0 if num_gpus == 1 else None
    
    # Tokenizer
    print(f"[TOPIC-EVAL] Loading tokenizer: {config.model.hf_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.hf_checkpoint,
        use_fast=True,
        padding_side='left'
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
    
    # Target singleton
    target_singleton = None
    target_tokenizer = None
    if config.environment.target_model:
        target_model_name = config.environment.target_model
        print(f"[TOPIC-EVAL] Loading target model: {target_model_name}")

        # Detect OpenAI-chat targets that should be accessed via API rather than HF Hub
        openai_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-5"]
        is_openai_target = any(target_model_name.startswith(m) for m in openai_models)

        if is_openai_target:
            print(f"[TOPIC-EVAL] OpenAI target model detected: {target_model_name}")
            print("[TOPIC-EVAL] Target model singleton and tokenizer will be handled by OpenAI API integration")
            target_singleton = None
            target_tokenizer = None
        else:
            target_quantize = getattr(config.environment, 'target_quantize', None)
            target_singleton = BaseModelSingleton.get_instance(
                model_name=target_model_name,
                use_offloading=config.training.use_offloading,
                config=config,
                prefer_gpu=target_prefer_gpu,
                quantize=target_quantize
            )
            target_tokenizer = AutoTokenizer.from_pretrained(
                target_model_name,
                use_fast=True,
                padding_side='left'
            )
            target_tokenizer.padding_side = 'left'
            if target_tokenizer.pad_token is None and target_tokenizer.eos_token is not None:
                target_tokenizer.pad_token = target_tokenizer.eos_token
                target_tokenizer.pad_token_id = target_tokenizer.eos_token_id
            target_singleton.tokenizer = target_tokenizer
    
    # Guard singleton
    guard_singleton = None
    guard_tokenizer = None
    if config.environment.guard_model:
        print(f"[TOPIC-EVAL] Loading guard model: {config.environment.guard_model}")
        guard_quantize = getattr(config.environment, 'guard_quantize', None)
        guard_singleton = BaseModelSingleton.get_instance(
            model_name=config.environment.guard_model,
            use_offloading=config.training.use_offloading,
            config=config,
            prefer_gpu=guard_prefer_gpu,
            quantize=guard_quantize
        )
        guard_tokenizer = AutoTokenizer.from_pretrained(
            config.environment.guard_model,
            use_fast=True,
            padding_side='left'
        )
        guard_tokenizer.padding_side = 'left'
        if guard_tokenizer.pad_token is None:
            if guard_tokenizer.eos_token is not None:
                guard_tokenizer.pad_token = guard_tokenizer.eos_token
                guard_tokenizer.pad_token_id = guard_tokenizer.eos_token_id
        guard_singleton.tokenizer = guard_tokenizer
    
    # Load policy and high-level policy
    print(f"[TOPIC-EVAL] Loading policy from: {policy_dir}")
    policy = _load_policy_with_singleton(policy_dir, None, device=device)
    
    encoder = None
    if hasattr(policy, 'base_model') and policy.base_model is not None:
        encoder = policy.base_model
    elif hasattr(policy, 'model') and policy.model is not None:
        encoder = policy.model
    
    high_level_policy = _load_high_level_policy(
        high_level_policy_dir,
        device=device,
        encoder=encoder,
        tokenizer=tokenizer
    )
    
    if high_level_policy is not None:
        # Don't replace tokenizer for freeform policies - they load their own from base model
        # Only replace for policies that need the main tokenizer (e.g., template-based policies)
        if not hasattr(high_level_policy, 'policy_kind') or high_level_policy.policy_kind != 'freeform':
            high_level_policy.tokenizer = tokenizer
        if hasattr(policy, 'model') and policy.model is not None:
            high_level_policy.encoder = policy.model
        high_level_policy = high_level_policy.to(device)
        print(f"[TOPIC-EVAL] Loaded high-level policy (type: {getattr(high_level_policy, 'policy_kind', 'unknown')})")
    else:
        print("[TOPIC-EVAL] No high-level policy found")

    sequence_critic = None
    critic_tokenizer = None
    use_hl_critic_selection = bool(
        getattr(config.model, "hl_critic_selection", False)
    )
    if use_hl_critic_selection:
        from src.critics import SequenceLevelValueCritic

        critic_root = critic_checkpoint_dir if critic_checkpoint_dir is not None else checkpoint_dir
        seq_dir = os.path.join(critic_root, "sequence_critic")
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(
                f"hl_critic_selection requires sequence_critic at {seq_dir}"
            )
        if high_level_policy is None or getattr(high_level_policy, "policy_kind", None) != "freeform":
            raise RuntimeError(
                "hl_critic_selection requires a loaded freeform high-level policy "
                "(shares Zephyr backbone with the critic)"
            )
        # Share HL Zephyr backbone; scoring disables LoRA adapters.
        critic_tokenizer = high_level_policy.tokenizer
        sequence_critic = SequenceLevelValueCritic(
            hf_checkpoint=getattr(
                high_level_policy.model.config,
                "_name_or_path",
                config.model.hf_checkpoint,
            ),
            tokenizer=critic_tokenizer,
            model=high_level_policy.model,
        )
        sequence_critic.load(seq_dir)
        sequence_critic.eval()
        print(f"[TOPIC-EVAL] Loaded HL selection critic from {seq_dir}")
    
    print("[TOPIC-EVAL] All models loaded successfully!\n")
    
    return {
        "policy": policy,
        "high_level_policy": high_level_policy,
        "sequence_critic": sequence_critic,
        "critic_tokenizer": critic_tokenizer,
        "target_singleton": target_singleton,
        "target_tokenizer": target_tokenizer,
        "guard_singleton": guard_singleton,
        "guard_tokenizer": guard_tokenizer,
        "tokenizer": tokenizer,
        "device": device
    }


def compute_baseline_asr(
    models: Dict[str, Any],
    config: AgentConfig,
    topic_name: str,
    topic_df: pd.DataFrame,
    num_samples: int = 20
) -> Dict[str, Any]:
    """
    Compute baseline ASR by sending original prompts directly to target model.
    
    Args:
        models: Dictionary of pre-loaded models
        config: Agent configuration
        topic_name: Name of the topic
        topic_df: DataFrame with topic prompts
        num_samples: Number of prompts to test
        
    Returns:
        Dictionary with baseline metrics
    """
    print(f"[TOPIC-EVAL] Computing baseline ASR for '{topic_name}'...")
    
    # Sample prompts
    sample_df = topic_df.sample(n=min(num_samples, len(topic_df)), replace=False)
    prompts = sample_df['Behavior'].tolist()
    
    # Create environment temporarily to use guard model
    hf_token = os.environ.get("HF_TOKEN") or None
    from src.environment import MultiStepEnv
    
    # Create minimal environment just for guard evaluation
    temp_config = AgentConfig.from_dict(config.__dict__)
    temp_env = MultiStepEnv(
        config=temp_config.environment,
        hf_token=hf_token,
        target_singleton=models["target_singleton"],
        target_tokenizer=models["target_tokenizer"],
        guard_singleton=models["guard_singleton"],
        guard_tokenizer=models["guard_tokenizer"],
        debug_mode=False,
        max_new_tokens=config.environment.max_new_tokens,
        target_quantize=getattr(config.environment, 'target_quantize', None),
        guard_quantize=getattr(config.environment, 'guard_quantize', None),
        evaluation_mode=True
    )
    
    # Get target responses for raw prompts
    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    responses = temp_env._get_target_responses(chats)
    
    # Evaluate with guard model (batch evaluation for efficiency)
    guard_scores_tensor = temp_env.score_safety(
        responses,
        questions=prompts,
        reference_state=prompts,
        include_white_box=False
    )
    
    # Convert to list and check successes
    guard_scores = [float(score) for score in guard_scores_tensor]
    successes = []
    harm_threshold = config.environment.harm_threshold
    
    for score in guard_scores:
        # Check if harmful (above threshold)
        is_harmful = score > harm_threshold
        successes.append(is_harmful)
    
    baseline_asr = sum(successes) / len(successes) if successes else 0.0
    baseline_harm_score = float(np.mean(guard_scores)) if guard_scores else 0.0
    
    print(f"[TOPIC-EVAL] Baseline ASR for '{topic_name}': {baseline_asr:.3f} ({sum(successes)}/{len(successes)})")
    
    return {
        "baseline_asr": float(baseline_asr),
        "baseline_harm_score": float(baseline_harm_score),
        "baseline_success_count": sum(successes),
        "baseline_total": len(successes),
        "baseline_guard_scores": guard_scores
    }


def evaluate_topic_with_loaded_models(
    models: Dict[str, Any],
    config: AgentConfig,
    topic_name: str,
    topic_df: pd.DataFrame,
    output_dir: str,
    eval_batch_size: int = 8,
    episodes: int = 20,
    max_queries: int = 30,
    trials_per_prompt: int = 1,
    compute_baseline: bool = True
) -> Dict[str, Any]:
    """
    Evaluate agent on a specific topic using pre-loaded models.
    
    Args:
        models: Dictionary of pre-loaded models from load_models_for_evaluation
        config: Agent configuration
        topic_name: Name of the topic category
        topic_df: DataFrame with topic-specific prompts
        output_dir: Directory to save results
        eval_batch_size: Batch size for evaluation
        episodes: Number of unique prompts to test
        max_queries: Maximum queries per episode
        trials_per_prompt: Number of times to repeat each prompt for variance estimation
        compute_baseline: Whether to compute baseline ASR (raw prompts, no agent)
        
    Returns:
        Dictionary with topic-specific metrics including per-episode data
    """
    
    print(f"\n[TOPIC-EVAL] ========================================")
    print(f"[TOPIC-EVAL] Evaluating on topic: {topic_name}")
    print(f"[TOPIC-EVAL] Dataset size: {len(topic_df)}")
    print(f"[TOPIC-EVAL] Unique prompts: {episodes}")
    print(f"[TOPIC-EVAL] Trials per prompt: {trials_per_prompt}")
    print(f"[TOPIC-EVAL] Total episodes: {episodes * trials_per_prompt}")
    print(f"[TOPIC-EVAL] Max queries: {max_queries}")
    print(f"[TOPIC-EVAL] ========================================\n")
    
    start_time = time.time()
    
    try:
        # Compute baseline ASR (raw prompts without agent)
        baseline_metrics = None
        if compute_baseline:
            baseline_metrics = compute_baseline_asr(
                models=models,
                config=config,
                topic_name=topic_name,
                topic_df=topic_df,
                num_samples=episodes
            )
            print(f"[TOPIC-EVAL] Baseline: ASR={baseline_metrics['baseline_asr']:.3f}, Harm Score={baseline_metrics['baseline_harm_score']:.3f}\n")
        # Create temporary dataset file for this topic
        topic_data_dir = os.path.join(output_dir, "topic_datasets")
        topic_file = create_topic_dataset_file(topic_df, topic_name, topic_data_dir)
        
        # Create environment with topic-specific data
        topic_config = AgentConfig.from_dict(config.__dict__)
        topic_config.environment.data_path = topic_file
        # Override batch_size to match number of episodes we want to evaluate
        topic_config.environment.batch_size = episodes
        
        hf_token = os.environ.get("HF_TOKEN") or None
        environment = MultiStepEnv(
            config=topic_config.environment,
            hf_token=hf_token,
            target_singleton=models["target_singleton"],
            target_tokenizer=models["target_tokenizer"],
            guard_singleton=models["guard_singleton"],
            guard_tokenizer=models["guard_tokenizer"],
            debug_mode=config.training.debug_mode,
            max_new_tokens=config.environment.max_new_tokens,
            target_quantize=getattr(config.environment, 'target_quantize', None),
            guard_quantize=getattr(config.environment, 'guard_quantize', None),
            evaluation_mode=True
        )
        
        # Attach environment to policy
        models["policy"].env = environment
        
        # Collect results for multiple trials
        all_success_list = []
        all_conversation_logs = []
        per_prompt_results = []  # Track results per unique prompt
        
        if trials_per_prompt > 1:
            # Sample unique prompts
            unique_prompts_df = topic_df.sample(n=min(episodes, len(topic_df)), replace=False)
            
            for trial in range(trials_per_prompt):
                print(f"[TOPIC-EVAL] Running trial {trial + 1}/{trials_per_prompt}...")
                
                # Re-save dataset with same prompts for consistent testing
                trial_topic_file = create_topic_dataset_file(unique_prompts_df, f"{topic_name}_trial{trial}", topic_data_dir)
                topic_config.environment.data_path = trial_topic_file
                # Ensure batch_size matches number of episodes
                topic_config.environment.batch_size = episodes
                
                # Recreate environment with same prompts
                environment = MultiStepEnv(
                    config=topic_config.environment,
                    hf_token=hf_token,
                    target_singleton=models["target_singleton"],
                    target_tokenizer=models["target_tokenizer"],
                    guard_singleton=models["guard_singleton"],
                    guard_tokenizer=models["guard_tokenizer"],
                    debug_mode=config.training.debug_mode,
                    max_new_tokens=config.environment.max_new_tokens,
                    target_quantize=getattr(config.environment, 'target_quantize', None),
                    guard_quantize=getattr(config.environment, 'guard_quantize', None),
                    evaluation_mode=True
                )
                models["policy"].env = environment
                
                success_list, conversation_logs = _evaluate_batch(
                    policy=models["policy"],
                    environment=environment,
                    max_queries=max_queries,
                    scenario="aware",
                    batch_size=eval_batch_size,
                    high_level_policy=models["high_level_policy"],
                    sequence_critic=models.get("sequence_critic"),
                    critic_tokenizer=models.get("critic_tokenizer"),
                    hl_selection_n_candidates=int(
                        getattr(config.model, "hl_selection_n_candidates", 1)
                    ),
                    hl_selection_score_head=str(
                        getattr(config.model, "hl_selection_score_head", "v")
                    ),
                )
                
                all_success_list.extend(success_list)
                all_conversation_logs.extend(conversation_logs)
            
            # Organize results by prompt
            for prompt_idx in range(episodes):
                prompt_trials = []
                for trial in range(trials_per_prompt):
                    result_idx = trial * episodes + prompt_idx
                    prompt_trials.append({
                        'success': all_success_list[result_idx],
                        'queries_used': all_conversation_logs[result_idx].get('queries_used', 0),
                        'harm_score': all_conversation_logs[result_idx].get('harm_score', 0.0)
                    })
                per_prompt_results.append(prompt_trials)
        else:
            # Single trial - standard evaluation
            print(f"[TOPIC-EVAL] Evaluating {episodes} episodes (single trial)...")
            all_success_list, all_conversation_logs = _evaluate_batch(
                policy=models["policy"],
                environment=environment,
                max_queries=max_queries,
                scenario="aware",
                batch_size=eval_batch_size,
                high_level_policy=models["high_level_policy"],
                sequence_critic=models.get("sequence_critic"),
                critic_tokenizer=models.get("critic_tokenizer"),
                hl_selection_n_candidates=int(
                    getattr(config.model, "hl_selection_n_candidates", 1)
                ),
                hl_selection_score_head=str(
                    getattr(config.model, "hl_selection_score_head", "v")
                ),
            )
        
        eval_time = time.time() - start_time
        
        # Compute aggregate metrics
        success_count = sum(all_success_list)
        total_episodes = len(all_success_list)
        asr = success_count / total_episodes if total_episodes > 0 else 0.0
        
        # Extract per-episode metrics
        episode_successes = [int(s) for s in all_success_list]
        queries_to_success = []
        queries_all = []
        harm_scores = []
        perplexities_all = []
        perplexities_success = []
        
        for log in all_conversation_logs:
            queries = log.get('queries_used', 0)
            queries_all.append(queries)
            
            # Compute harm_score from conversation turns
            # For successful episodes, use guard_score from the successful turn
            # For unsuccessful episodes, use max guard_score across all turns
            harm_score = log.get('harm_score', None)
            if harm_score is None:
                # Extract guard_scores from conversation turns
                conversation = log.get('conversation', [])
                if conversation:
                    turn_guard_scores = [turn.get('guard_score', 0.0) for turn in conversation if 'guard_score' in turn]
                    if turn_guard_scores:
                        if log.get('success', False):
                            # For successful episodes, use guard_score from the turn that succeeded
                            # Find the first turn where guard_unsafe is True
                            for turn in conversation:
                                if turn.get('guard_unsafe', False) and 'guard_score' in turn:
                                    harm_score = turn['guard_score']
                                    break
                            # Fallback: use max if we couldn't find the successful turn
                            if harm_score is None:
                                harm_score = max(turn_guard_scores)
                        else:
                            # For unsuccessful episodes, use max guard_score
                            harm_score = max(turn_guard_scores)
                    else:
                        harm_score = 0.0
                else:
                    harm_score = 0.0
            
            harm_scores.append(harm_score)
            
            # Get perplexity if available
            avg_ppl = log.get('avg_perplexity')
            if avg_ppl is not None:
                perplexities_all.append(avg_ppl)
                if log.get('success', False):
                    perplexities_success.append(avg_ppl)
            
            # Only count queries for successful episodes
            if log.get('success', False):
                queries_to_success.append(queries)
        
        # Compute variance metrics
        queries_to_success_mean = float(np.mean(queries_to_success)) if queries_to_success else 0.0
        queries_to_success_std = float(np.std(queries_to_success)) if len(queries_to_success) > 1 else 0.0
        
        # Compute perplexity metrics
        avg_perplexity_all = float(np.mean(perplexities_all)) if perplexities_all else None
        avg_perplexity_success = float(np.mean(perplexities_success)) if perplexities_success else None
        perplexity_std = float(np.std(perplexities_all)) if len(perplexities_all) > 1 else 0.0
        
        metrics = {
            "topic": topic_name,
            "unique_prompts": episodes,
            "trials_per_prompt": trials_per_prompt,
            "episodes": total_episodes,
            "success_count": success_count,
            "total_count": total_episodes,
            "asr": float(asr),
            "avg_harm_score": float(np.mean(harm_scores)) if harm_scores else 0.0,
            "avg_conversation_length": float(np.mean(queries_all)) if queries_all else 0.0,
            "queries_to_success_mean": queries_to_success_mean,
            "queries_to_success_std": queries_to_success_std,
            "queries_to_success_min": float(np.min(queries_to_success)) if queries_to_success else 0.0,
            "queries_to_success_max": float(np.max(queries_to_success)) if queries_to_success else 0.0,
            "first_turn_success_rate": sum(1 for q in queries_to_success if q == 1) / len(queries_to_success) if queries_to_success else 0.0,
            "avg_perplexity": avg_perplexity_all,
            "avg_perplexity_success": avg_perplexity_success,
            "perplexity_std": perplexity_std,
            "eval_time": float(eval_time),
            # Individual episode data for variance analysis
            "episode_successes": episode_successes,
            "episode_queries": queries_all,
            "episode_harm_scores": [float(h) for h in harm_scores],
            "episode_perplexities": perplexities_all if perplexities_all else []
        }
        
        # Add baseline metrics if computed
        if baseline_metrics is not None:
            metrics["baseline_asr"] = baseline_metrics["baseline_asr"]
            metrics["baseline_harm_score"] = baseline_metrics["baseline_harm_score"]
            metrics["baseline_success_count"] = baseline_metrics["baseline_success_count"]
            metrics["baseline_total"] = baseline_metrics["baseline_total"]
            metrics["asr_improvement"] = float(asr - baseline_metrics["baseline_asr"])
            metrics["episode_baseline_scores"] = baseline_metrics["baseline_guard_scores"]
        
        # Add per-prompt variance data if multiple trials
        if trials_per_prompt > 1 and per_prompt_results:
            prompt_asrs = []
            prompt_query_means = []
            for prompt_trials in per_prompt_results:
                prompt_asr = sum(t['success'] for t in prompt_trials) / len(prompt_trials)
                prompt_asrs.append(prompt_asr)
                successful_queries = [t['queries_used'] for t in prompt_trials if t['success']]
                if successful_queries:
                    prompt_query_means.append(np.mean(successful_queries))
            
            metrics["per_prompt_asr_mean"] = float(np.mean(prompt_asrs)) if prompt_asrs else 0.0
            metrics["per_prompt_asr_std"] = float(np.std(prompt_asrs)) if len(prompt_asrs) > 1 else 0.0
            metrics["per_prompt_data"] = per_prompt_results
        
        # Compute diversity and complexity metrics per category
        print(f"[TOPIC-EVAL] Computing diversity and complexity metrics for topic '{topic_name}'...")
        try:
            # Add topic name to each log for category grouping
            logs_with_category = []
            for log in all_conversation_logs:
                log_copy = log.copy()
                log_copy['topic'] = topic_name
                logs_with_category.append(log_copy)
            
            diversity_metrics = compute_diversity_and_complexity_per_category(
                logs_with_category,
                category_column='topic'
            )
            
            if topic_name in diversity_metrics:
                topic_diversity = diversity_metrics[topic_name]
                metrics["agent_diversity"] = topic_diversity["agent_diversity"]
                metrics["user_diversity"] = topic_diversity["user_diversity"]
                metrics["agent_complexity"] = topic_diversity["agent_complexity"]
                metrics["num_agent_messages"] = topic_diversity["num_agent_messages"]
                metrics["num_user_messages"] = topic_diversity["num_user_messages"]
                
                print(f"  Agent diversity (mean cosine distance): {metrics['agent_diversity']:.4f}")
                print(f"  User diversity (mean cosine distance): {metrics['user_diversity']:.4f}")
                print(f"  Agent complexity - mean length: {metrics['agent_complexity']['mean_length']:.1f}")
                print(f"  Agent complexity - vocab diversity: {metrics['agent_complexity']['vocab_diversity']:.4f}")
        except Exception as e:
            print(f"[TOPIC-EVAL] Warning: Failed to compute diversity metrics: {e}")
            import traceback
            traceback.print_exc()
            # Add placeholder metrics
            metrics["agent_diversity"] = None
            metrics["user_diversity"] = None
            metrics["agent_complexity"] = None
        
        # Save topic-specific results
        topic_output_dir = os.path.join(output_dir, topic_name)
        os.makedirs(topic_output_dir, exist_ok=True)
        
        metrics_path = os.path.join(topic_output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save conversation logs
        conv_log_path = os.path.join(topic_output_dir, "conversations.jsonl")
        with open(conv_log_path, 'w') as f:
            for log in all_conversation_logs:
                f.write(json.dumps(log) + "\n")
        
        print(f"\n[TOPIC-EVAL] Topic '{topic_name}' complete:")
        if baseline_metrics is not None:
            print(f"  Baseline ASR: {baseline_metrics['baseline_asr']:.3f}")
            print(f"  Agent ASR: {asr:.3f} ({success_count}/{total_episodes})")
            print(f"  Improvement: {metrics['asr_improvement']:.3f} ({metrics['asr_improvement']/baseline_metrics['baseline_asr']*100:.1f}% relative)" if baseline_metrics['baseline_asr'] > 0 else f"  Improvement: {metrics['asr_improvement']:.3f}")
        else:
            print(f"  ASR: {asr:.3f} ({success_count}/{total_episodes})")
        print(f"  Queries to success: {queries_to_success_mean:.2f} ± {queries_to_success_std:.2f}")
        print(f"  Avg harm score: {metrics['avg_harm_score']:.3f}")
        if avg_perplexity_all is not None:
            print(f"  Avg perplexity: {avg_perplexity_all:.2f} ± {perplexity_std:.2f}")
            if avg_perplexity_success is not None:
                print(f"  Avg perplexity (successful): {avg_perplexity_success:.2f}")
        if trials_per_prompt > 1:
            print(f"  Per-prompt ASR std: {metrics.get('per_prompt_asr_std', 0):.3f}")
        print(f"  Time: {eval_time:.1f}s")

        # Print one full example conversation for this topic after evaluation concludes.
        # Prefer a successful attack episode if available; otherwise, use the first episode.
        example_log = None
        for log in all_conversation_logs:
            if log.get("success", False):
                example_log = log
                break
        if example_log is None and all_conversation_logs:
            example_log = all_conversation_logs[0]

        if example_log is not None:
            print(f"\n[TOPIC-EVAL] Example conversation for topic '{topic_name}':")
            print(json.dumps(example_log, indent=2))
        
        return metrics
        
    except Exception as e:
        print(f"[TOPIC-EVAL] ERROR evaluating topic '{topic_name}': {e}")
        import traceback
        traceback.print_exc()
        raise e
        


def create_topic_config(base_config: AgentConfig, topic_data_path: str, episodes: int, max_queries: int) -> Dict[str, Any]:
    """
    Create topic-specific configuration by modifying base config.
    
    Args:
        base_config: Base agent configuration
        topic_data_path: Path to topic-specific dataset
        episodes: Number of episodes
        max_queries: Maximum queries per episode
        
    Returns:
        Dictionary with topic-specific config
    """
    # Convert config to dict
    config_dict = {
        "training": {
            "episodes": episodes,
            "batch_size": base_config.training.batch_size,
            "rollout_steps": base_config.training.rollout_steps
        },
        "model": {
            "model_name": base_config.model.model_name,
            "hf_checkpoint": base_config.model.hf_checkpoint,
            "use_high_level_policy": base_config.model.use_high_level_policy,
            "high_level_policy_type": getattr(base_config.model, 'high_level_policy_type', 'ppo'),
            "templates_path": getattr(base_config.model, 'templates_path', 'src/configs/templates.json'),
            "use_lora": base_config.model.use_lora,
            "lora_r": base_config.model.lora_r,
            "lora_alpha": base_config.model.lora_alpha,
            "target_modules": base_config.model.target_modules,
            "target_layers": base_config.model.target_layers
        },
        "environment": {
            "data_path": topic_data_path,
            "harm_threshold": base_config.environment.harm_threshold,
            "max_steps": base_config.environment.max_steps,
            "batch_size": base_config.environment.batch_size,
            "max_new_tokens": base_config.environment.max_new_tokens,
            "target_model": base_config.environment.target_model,
            "guard_model": base_config.environment.guard_model,
            "target_quantize": getattr(base_config.environment, 'target_quantize', False),
            "guard_quantize": getattr(base_config.environment, 'guard_quantize', False)
        },
        "episodes": episodes,
        "max_queries": max_queries
    }
    
    return config_dict


def compute_topic_metrics(
    results: Dict[str, List[bool]],
    topic_name: str,
    episodes: int,
    eval_time: float,
    output_dir: str
) -> Dict[str, Any]:
    """
    Compute topic-specific metrics from evaluation results.
    
    Args:
        results: Dictionary mapping test names to success lists
        topic_name: Name of the topic
        episodes: Number of episodes run
        eval_time: Time taken for evaluation
        output_dir: Directory with conversation logs
        
    Returns:
        Dictionary with computed metrics
    """
    # Aggregate results across all test configs
    all_successes = []
    for test_name, success_list in results.items():
        all_successes.extend(success_list)
    
    success_count = sum(all_successes)
    total_count = len(all_successes)
    asr = success_count / total_count if total_count > 0 else 0.0
    
    # Try to compute additional metrics from conversation logs
    avg_harm_score = 0.0
    avg_conversation_length = 0.0
    first_turn_success_rate = 0.0
    
    # Look for conversation log files in output_dir
    conversation_files = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.jsonl') and 'conversation' in file:
                conversation_files.append(os.path.join(output_dir, file))
    
    if conversation_files:
        harm_scores = []
        conversation_lengths = []
        first_turn_successes = 0
        
        for conv_file in conversation_files:
            try:
                with open(conv_file, 'r') as f:
                    for line in f:
                        conv = json.loads(line)
                        
                        # Extract harm score if available
                        if 'harm_score' in conv:
                            harm_scores.append(conv['harm_score'])
                        elif 'guard_score' in conv:
                            harm_scores.append(conv['guard_score'])
                        
                        # Extract conversation length
                        if 'conversation' in conv:
                            conversation_lengths.append(len(conv['conversation']))
                        elif 'turns' in conv:
                            conversation_lengths.append(conv['turns'])
                        
                        # Check first turn success
                        if 'first_turn_success' in conv:
                            first_turn_successes += int(conv['first_turn_success'])
                        elif conv.get('success', False) and len(conv.get('conversation', [])) == 1:
                            first_turn_successes += 1
            except Exception as e:
                print(f"[TOPIC-EVAL] Warning: Could not parse conversation file {conv_file}: {e}")
        
        if harm_scores:
            avg_harm_score = np.mean(harm_scores)
        if conversation_lengths:
            avg_conversation_length = np.mean(conversation_lengths)
        if total_count > 0:
            first_turn_success_rate = first_turn_successes / total_count
    
    metrics = {
        "topic": topic_name,
        "episodes": episodes,
        "success_count": success_count,
        "total_count": total_count,
        "asr": float(asr),
        "avg_harm_score": float(avg_harm_score),
        "avg_conversation_length": float(avg_conversation_length),
        "first_turn_success_rate": float(first_turn_success_rate),
        "eval_time": float(eval_time)
    }
    
    return metrics


def run_topic_level_evaluation(
    config_path: str,
    checkpoint_dir: str,
    output_dir: str,
    eval_batch_size: int = 8,
    episodes_per_topic: int = 20,
    max_queries: int = 30,
    specific_topics: Optional[List[str]] = None,
    trials_per_prompt: int = 1,
    critic_checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full topic-level evaluation across all harm categories.
    
    Args:
        config_path: Path to base configuration file
        checkpoint_dir: Path to model checkpoint
        output_dir: Directory to save all results
        eval_batch_size: Batch size for evaluation
        episodes_per_topic: Episodes to run per topic
        max_queries: Maximum queries per episode
        specific_topics: Optional list of specific topics to evaluate (None = all)
        critic_checkpoint_dir: Optional final_model with sequence_critic for HL selection
        
    Returns:
        Dictionary with aggregated results across all topics
    """
    print(f"\n[TOPIC-EVAL] ============================================================")
    print(f"[TOPIC-EVAL] STARTING TOPIC-LEVEL EVALUATION")
    print(f"[TOPIC-EVAL] ============================================================")
    print(f"[TOPIC-EVAL] Config: {config_path}")
    print(f"[TOPIC-EVAL] Checkpoint: {checkpoint_dir}")
    print(f"[TOPIC-EVAL] Critic checkpoint: {critic_checkpoint_dir or checkpoint_dir}")
    print(f"[TOPIC-EVAL] Output: {output_dir}")
    print(f"[TOPIC-EVAL] Episodes per topic: {episodes_per_topic}")
    print(f"[TOPIC-EVAL] Max queries: {max_queries}")
    print(f"[TOPIC-EVAL] ============================================================\n")
    
    # Load and clean configuration
    cfg_dict = _load_json(config_path)
    cleaned_cfg_dict = _clean_config_for_evaluation(cfg_dict)
    config = AgentConfig.from_dict(cleaned_cfg_dict)
    
    # Get HarmBench path from config
    harmbench_path = config.environment.data_path
    if not harmbench_path.endswith('.csv'):
        # If directory provided, look for the main file
        if os.path.isdir(harmbench_path):
            harmbench_path = os.path.join(harmbench_path, "harmbench_behaviors_text_all.csv")
    
    if not os.path.exists(harmbench_path):
        _fail(f"HarmBench dataset not found: {harmbench_path}")
    
    # Load and partition dataset by topic
    topics = load_and_partition_by_topic(harmbench_path, category_column="SemanticCategory")
    
    # Filter to specific topics if requested
    if specific_topics:
        topics = {k: v for k, v in topics.items() if k in specific_topics}
        print(f"\n[TOPIC-EVAL] Filtered to specific topics: {list(topics.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save evaluation configuration
    eval_config = {
        "checkpoint_dir": checkpoint_dir,
        "critic_checkpoint_dir": critic_checkpoint_dir or checkpoint_dir,
        "config_path": config_path,
        "episodes_per_topic": episodes_per_topic,
        "max_queries": max_queries,
        "eval_batch_size": eval_batch_size,
        "trials_per_prompt": trials_per_prompt,
        "hl_critic_selection": bool(getattr(config.model, "hl_critic_selection", False)),
        "hl_selection_n_candidates": int(getattr(config.model, "hl_selection_n_candidates", 1)),
        "hl_selection_score_head": str(getattr(config.model, "hl_selection_score_head", "v")),
        "topics_evaluated": list(topics.keys()),
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "eval_config.json"), 'w') as f:
        json.dump(eval_config, f, indent=2)
    
    # Load models once for all topics
    models = load_models_for_evaluation(
        config, checkpoint_dir, critic_checkpoint_dir=critic_checkpoint_dir
    )
    
    # Run evaluation on each topic
    topic_results = {}
    start_time = time.time()
    
    for idx, (topic_name, topic_df) in enumerate(topics.items(), 1):
        print(f"\n[TOPIC-EVAL] ============================================================")
        print(f"[TOPIC-EVAL] Topic {idx}/{len(topics)}: {topic_name}")
        print(f"[TOPIC-EVAL] ============================================================")
        
        # Adjust episodes if topic is too small
        topic_episodes = min(episodes_per_topic, len(topic_df))
        if topic_episodes < episodes_per_topic:
            print(f"[TOPIC-EVAL] WARNING: Topic has only {len(topic_df)} prompts, running {topic_episodes} episodes")
        
        metrics = evaluate_topic_with_loaded_models(
            models=models,
            config=config,
            topic_name=topic_name,
            topic_df=topic_df,
            output_dir=output_dir,
            eval_batch_size=eval_batch_size,
            episodes=topic_episodes,
            max_queries=max_queries,
            trials_per_prompt=trials_per_prompt
        )
        
        topic_results[topic_name] = metrics
    
    total_time = time.time() - start_time
    
    # Compute aggregated statistics
    summary = compute_summary_statistics(topic_results, total_time)
    
    # Save results
    save_results(output_dir, topic_results, summary)
    
    # Print final summary
    print_summary(summary)
    
    return summary


def compute_summary_statistics(topic_results: Dict[str, Dict[str, Any]], total_time: float) -> Dict[str, Any]:
    """
    Compute aggregated statistics across all topics.
    
    Aggregates metrics from all episodes across all topics, not just averaging
    per-topic means, to get accurate overall statistics.
    
    Args:
        topic_results: Dictionary mapping topic names to metrics
        total_time: Total evaluation time
        
    Returns:
        Dictionary with summary statistics
    """
    # Extract per-topic metrics for topic-level statistics
    asrs = [m['asr'] for m in topic_results.values() if 'asr' in m]
    harm_scores = [m['avg_harm_score'] for m in topic_results.values() if 'avg_harm_score' in m]
    conv_lengths = [m['avg_conversation_length'] for m in topic_results.values() if 'avg_conversation_length' in m]
    
    # Aggregate across all episodes from all topics
    total_successes = sum(m['success_count'] for m in topic_results.values())
    total_episodes = sum(m['episodes'] for m in topic_results.values())
    
    # Collect all episode-level data across all topics
    all_episode_queries = []
    all_episode_successes = []
    all_episode_harm_scores = []
    all_episode_perplexities = []
    all_queries_to_success = []  # Only successful episodes
    
    for topic_metrics in topic_results.values():
        # Collect queries for all episodes
        if 'episode_queries' in topic_metrics:
            all_episode_queries.extend(topic_metrics['episode_queries'])
        
        # Collect success indicators
        if 'episode_successes' in topic_metrics:
            all_episode_successes.extend(topic_metrics['episode_successes'])
        
        # Collect harm scores for all episodes
        if 'episode_harm_scores' in topic_metrics:
            all_episode_harm_scores.extend(topic_metrics['episode_harm_scores'])
        
        # Collect perplexities for all episodes
        if 'episode_perplexities' in topic_metrics and topic_metrics['episode_perplexities']:
            all_episode_perplexities.extend(topic_metrics['episode_perplexities'])
    
    # Compute time-to-harm (queries to success) from all successful episodes
    if all_episode_queries and all_episode_successes:
        for queries, success in zip(all_episode_queries, all_episode_successes):
            if success:
                all_queries_to_success.append(queries)
    
    # Compute overall statistics
    overall_queries_to_success_mean = float(np.mean(all_queries_to_success)) if all_queries_to_success else 0.0
    overall_queries_to_success_std = float(np.std(all_queries_to_success)) if len(all_queries_to_success) > 1 else 0.0
    overall_queries_to_success_min = float(np.min(all_queries_to_success)) if all_queries_to_success else 0.0
    overall_queries_to_success_max = float(np.max(all_queries_to_success)) if all_queries_to_success else 0.0
    overall_first_turn_success_rate = sum(1 for q in all_queries_to_success if q == 1) / len(all_queries_to_success) if all_queries_to_success else 0.0
    
    # Overall harm score from all episodes
    overall_harm_score_mean = float(np.mean(all_episode_harm_scores)) if all_episode_harm_scores else 0.0
    overall_harm_score_std = float(np.std(all_episode_harm_scores)) if len(all_episode_harm_scores) > 1 else 0.0
    
    # Overall conversation length from all episodes
    overall_conv_length_mean = float(np.mean(all_episode_queries)) if all_episode_queries else 0.0
    overall_conv_length_std = float(np.std(all_episode_queries)) if len(all_episode_queries) > 1 else 0.0
    
    # Overall perplexity statistics
    overall_perplexity_mean = float(np.mean(all_episode_perplexities)) if all_episode_perplexities else None
    overall_perplexity_std = float(np.std(all_episode_perplexities)) if len(all_episode_perplexities) > 1 else None
    # Perplexity for successful episodes only
    successful_perplexities = []
    if all_episode_perplexities and all_episode_successes:
        for ppl, success in zip(all_episode_perplexities, all_episode_successes):
            if success and ppl is not None:
                successful_perplexities.append(ppl)
    overall_perplexity_success_mean = float(np.mean(successful_perplexities)) if successful_perplexities else None
    
    summary = {
        "total_topics": len(topic_results),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_asr": total_successes / total_episodes if total_episodes > 0 else 0.0,
        
        # Per-topic ASR statistics (for variance across topics)
        "asr_mean": float(np.mean(asrs)) if asrs else 0.0,
        "asr_std": float(np.std(asrs)) if asrs else 0.0,
        "asr_min": float(np.min(asrs)) if asrs else 0.0,
        "asr_max": float(np.max(asrs)) if asrs else 0.0,
        
        # Overall harm score (aggregated from all episodes)
        "overall_harm_score_mean": overall_harm_score_mean,
        "overall_harm_score_std": overall_harm_score_std,
        # Per-topic harm score statistics (for variance across topics)
        "avg_harm_score_mean": float(np.mean(harm_scores)) if harm_scores else 0.0,
        "avg_harm_score_std": float(np.std(harm_scores)) if harm_scores else 0.0,
        
        # Overall conversation length (aggregated from all episodes)
        "overall_conversation_length_mean": overall_conv_length_mean,
        "overall_conversation_length_std": overall_conv_length_std,
        # Per-topic conversation length statistics (for variance across topics)
        "avg_conversation_length_mean": float(np.mean(conv_lengths)) if conv_lengths else 0.0,
        "avg_conversation_length_std": float(np.std(conv_lengths)) if conv_lengths else 0.0,
        
        # Overall time-to-harm (aggregated from all successful episodes)
        "overall_queries_to_success_mean": overall_queries_to_success_mean,
        "overall_queries_to_success_std": overall_queries_to_success_std,
        "overall_queries_to_success_min": overall_queries_to_success_min,
        "overall_queries_to_success_max": overall_queries_to_success_max,
        "overall_first_turn_success_rate": overall_first_turn_success_rate,
        
        # Overall perplexity (aggregated from all episodes)
        "overall_perplexity_mean": overall_perplexity_mean,
        "overall_perplexity_std": overall_perplexity_std,
        "overall_perplexity_success_mean": overall_perplexity_success_mean,
        
        "total_eval_time": float(total_time),
        "timestamp": datetime.now().isoformat(),
        "topics": topic_results
    }
    
    return summary


def save_results(output_dir: str, topic_results: Dict[str, Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """
    Save evaluation results to files.
    
    Args:
        output_dir: Directory to save results
        topic_results: Per-topic metrics
        summary: Aggregated summary statistics
    """
    # Save summary JSON
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[TOPIC-EVAL] Saved summary to: {summary_path}")
    
    # Save results as CSV
    csv_path = os.path.join(output_dir, "results_by_topic.csv")
    df = pd.DataFrame([
        {
            'topic': topic,
            'episodes': metrics['episodes'],
            'successes': metrics['success_count'],
            'total': metrics.get('total_count', metrics['episodes']),
            'asr': metrics['asr'],
            'avg_harm_score': metrics['avg_harm_score'],
            'avg_conversation_length': metrics['avg_conversation_length'],
            'queries_to_success_mean': metrics.get('queries_to_success_mean', 0.0),
            'queries_to_success_std': metrics.get('queries_to_success_std', 0.0),
            'first_turn_success_rate': metrics.get('first_turn_success_rate', 0.0),
            'eval_time_seconds': metrics['eval_time']
        }
        for topic, metrics in topic_results.items()
    ])
    df = df.sort_values('asr', ascending=False)
    df.to_csv(csv_path, index=False)
    print(f"[TOPIC-EVAL] Saved CSV results to: {csv_path}")


def print_summary(summary: Dict[str, Any]) -> None:
    """
    Print human-readable summary of results.
    
    Args:
        summary: Summary statistics dictionary
    """
    print(f"\n[TOPIC-EVAL] ============================================================")
    print(f"[TOPIC-EVAL] EVALUATION COMPLETE")
    print(f"[TOPIC-EVAL] ============================================================")
    print(f"[TOPIC-EVAL] Total topics evaluated: {summary['total_topics']}")
    print(f"[TOPIC-EVAL] Total episodes: {summary['total_episodes']}")
    print(f"[TOPIC-EVAL] Total successes: {summary['total_successes']}")
    print(f"[TOPIC-EVAL] Overall ASR: {summary['overall_asr']:.3f}")
    print(f"\n[TOPIC-EVAL] --- Overall Statistics (Aggregated from All Episodes) ---")
    print(f"[TOPIC-EVAL] Overall harm score: {summary['overall_harm_score_mean']:.3f} ± {summary['overall_harm_score_std']:.3f}")
    print(f"[TOPIC-EVAL] Overall conversation length: {summary['overall_conversation_length_mean']:.2f} ± {summary['overall_conversation_length_std']:.2f}")
    print(f"[TOPIC-EVAL] Overall time-to-harm (queries to success): {summary['overall_queries_to_success_mean']:.2f} ± {summary['overall_queries_to_success_std']:.2f}")
    print(f"[TOPIC-EVAL]   Range: [{summary['overall_queries_to_success_min']:.0f}, {summary['overall_queries_to_success_max']:.0f}]")
    print(f"[TOPIC-EVAL] Overall first-turn success rate: {summary['overall_first_turn_success_rate']:.3f}")
    if summary.get('overall_perplexity_mean') is not None:
        print(f"[TOPIC-EVAL] Overall perplexity: {summary['overall_perplexity_mean']:.2f} ± {summary.get('overall_perplexity_std', 0.0):.2f}")
        if summary.get('overall_perplexity_success_mean') is not None:
            print(f"[TOPIC-EVAL] Overall perplexity (successful episodes): {summary['overall_perplexity_success_mean']:.2f}")
    print(f"\n[TOPIC-EVAL] --- Per-Topic Statistics (Variance Across Topics) ---")
    print(f"[TOPIC-EVAL] ASR statistics: {summary['asr_mean']:.3f} ± {summary['asr_std']:.3f} (min: {summary['asr_min']:.3f}, max: {summary['asr_max']:.3f})")
    print(f"[TOPIC-EVAL] Avg harm score (per-topic): {summary['avg_harm_score_mean']:.3f} ± {summary['avg_harm_score_std']:.3f}")
    print(f"[TOPIC-EVAL] Avg conversation length (per-topic): {summary['avg_conversation_length_mean']:.2f} ± {summary['avg_conversation_length_std']:.2f}")
    print(f"[TOPIC-EVAL] Total evaluation time: {summary['total_eval_time']:.1f}s ({summary['total_eval_time']/60:.1f}m)")
    print(f"\n[TOPIC-EVAL] Per-topic results:")
    
    # Sort topics by ASR
    sorted_topics = sorted(summary['topics'].items(), key=lambda x: x[1]['asr'], reverse=True)
    for topic, metrics in sorted_topics:
        queries_to_success = metrics.get('queries_to_success_mean', 0.0)
        print(f"  {topic:30s}: ASR={metrics['asr']:.3f}  harm={metrics['avg_harm_score']:.3f}  len={metrics['avg_conversation_length']:.1f}  time-to-harm={queries_to_success:.1f}")
    
    print(f"[TOPIC-EVAL] ============================================================\n")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    """Main entry point for topic-level evaluation."""
    parser = argparse.ArgumentParser(description="Topic-level evaluation of red-teaming agents")
    parser.add_argument("--config-path", type=str, required=True, 
                       help="Path to base configuration file")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save evaluation results")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                       help="Batch size for evaluation (default: 8)")
    parser.add_argument("--episodes-per-topic", type=int, default=20,
                       help="Number of episodes to run per topic (default: 20)")
    parser.add_argument("--max-queries", type=int, default=30,
                       help="Maximum queries per episode (default: 30)")
    parser.add_argument("--topics", type=str, nargs='+', default=None,
                       help="Specific topics to evaluate (default: all)")
    parser.add_argument("--trials-per-prompt", type=int, default=1,
                       help="Number of times to repeat each prompt for variance estimation (default: 1)")
    parser.add_argument(
        "--critic-checkpoint-dir",
        type=str,
        default=None,
        help="final_model dir whose sequence_critic/ is used for HL strategy selection "
             "(defaults to --checkpoint-dir)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.config_path):
        _fail(f"Configuration file not found: {args.config_path}")
    if not os.path.exists(args.checkpoint_dir):
        _fail(f"Checkpoint directory not found: {args.checkpoint_dir}")
    if args.critic_checkpoint_dir is not None and not os.path.exists(args.critic_checkpoint_dir):
        _fail(f"Critic checkpoint directory not found: {args.critic_checkpoint_dir}")
    
    # Run evaluation
    summary = run_topic_level_evaluation(
        config_path=args.config_path,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        eval_batch_size=args.eval_batch_size,
        episodes_per_topic=args.episodes_per_topic,
        max_queries=args.max_queries,
        specific_topics=args.topics,
        trials_per_prompt=args.trials_per_prompt,
        critic_checkpoint_dir=args.critic_checkpoint_dir,
    )
    
    print(f"[TOPIC-EVAL] Results saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

