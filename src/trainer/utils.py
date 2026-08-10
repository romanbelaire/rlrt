"""
Trainer utility helpers shared between AgentTrainer modules.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_gpu_memory(self_or_none=None, prefix: str = "", reset_peak: bool = False, flush: bool = False) -> None:
    """Print a one-line summary of current / peak GPU memory usage.

    This is only intended for lightweight debugging in *debug_mode*.
    It purposefully fails fast if CUDA is unavailable so that incorrect
    assumptions about the execution environment surface immediately.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("_log_gpu_memory() called but CUDA is not available")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(
            f"[GPU MEMORY] {prefix} | allocated: {allocated_gb:.2f} GB | "
            f"peak: {peak_gb:.2f} GB | reserved: {reserved_gb:.2f} GB",
            flush=flush,
        )
    else:
        print(f"[GPU MEMORY] {prefix} | Multi-GPU ({num_gpus} devices):")
        for device_id in range(num_gpus):
            with torch.cuda.device(device_id):
                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                total_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
                print(
                    f"  GPU {device_id}: allocated: {allocated_gb:.2f} GB | "
                    f"peak: {peak_gb:.2f} GB | reserved: {reserved_gb:.2f} GB | "
                    f"total: {total_gb:.2f} GB",
                    flush=flush,
                )

    if reset_peak:
        torch.cuda.reset_peak_memory_stats()


def _reset_gpu_memory_stats(self_or_none=None) -> None:
    """Reset GPU memory peak statistics (optionally logging when in debug mode)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        if (
            self_or_none
            and hasattr(self_or_none, "config")
            and getattr(self_or_none.config.training, "debug_mode", False)
        ):
            print("[GPU MEMORY] Reset peak memory statistics")


def get_optimal_topk_for_kl_divergence(
    vocab_size: int, available_memory_gb: float, batch_size: int, seq_len: int
) -> int:
    """Estimate a safe top-k size for KL calculations based on available GPU memory."""
    memory_per_token = 8 * 2  # 8 bytes per logit, two models
    total_tokens = batch_size * seq_len
    usable_memory_bytes = available_memory_gb * 1024**3 * 0.5  # use only half to stay safe
    max_tokens_in_memory = usable_memory_bytes / memory_per_token
    max_topk = min(vocab_size, int(max_tokens_in_memory / total_tokens))
    return max(100, min(max_topk, 5000))


def get_base_model_from_policy(policy: nn.Module) -> Optional[nn.Module]:
    """Return the underlying base model for PEFT/LoRA-wrapped policies, if any."""
    if hasattr(policy, "base_model"):
        return policy.base_model
    if hasattr(policy, "model") and hasattr(policy.model, "base_model"):
        return policy.model.base_model
    return None


def compute_kl_divergence_from_base_model(
    policy: nn.Module,
    base_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kl_coef: float = 0.1,
    minibatch_size: int = 8,
    device: str = "cuda",
    top_k: int = 1000,
) -> torch.Tensor:
    """Compute KL(policy || base_model) using only the base model's top-k tokens.
    
    Handles models on different devices (e.g., policy on cuda:0, base_model on cuda:1).
    """
    # Determine actual devices for each model
    try:
        base_model_device = next(base_model.parameters()).device
    except StopIteration:
        # Fallback: try to get device from model state dict
        if base_model.state_dict():
            base_model_device = next(iter(base_model.state_dict().values())).device
        else:
            base_model_device = torch.device(device)
    
    try:
        policy_device = next(policy.parameters()).device
    except StopIteration:
        # Fallback: try to get device from model state dict
        if policy.state_dict():
            policy_device = next(iter(policy.state_dict().values())).device
        else:
            policy_device = torch.device(device)
    
    batch_size = input_ids.shape[0]
    total_kl_loss = 0.0
    num_minibatches = 0

    for i in range(0, batch_size, minibatch_size):
        end_idx = min(i + minibatch_size, batch_size)
        mb_input_ids = input_ids[i:end_idx]
        mb_attention_mask = attention_mask[i:end_idx]

        # Move inputs to base_model's device for base_model forward pass
        mb_input_ids_base = mb_input_ids.to(base_model_device)
        mb_attention_mask_base = mb_attention_mask.to(base_model_device)

        with torch.no_grad():
            base_outputs = base_model(
                input_ids=mb_input_ids_base,
                attention_mask=mb_attention_mask_base,
                output_hidden_states=False,
            )
            base_probs = F.softmax(base_outputs.logits, dim=-1)
            _, base_topk_indices = torch.topk(base_probs, top_k, dim=-1)
            # Extract top-k probabilities immediately to save memory
            base_probs_sub = torch.gather(base_probs, dim=-1, index=base_topk_indices)
            # Move to CPU to free GPU memory on base_model_device
            base_topk_indices_cpu = base_topk_indices.cpu()
            base_probs_sub_cpu = base_probs_sub.cpu()
            del base_outputs, base_probs, base_probs_sub  # Free memory immediately

        # Clear GPU cache after base model forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Move inputs to policy's device for policy forward pass
        mb_input_ids_policy = mb_input_ids.to(policy_device)
        mb_attention_mask_policy = mb_attention_mask.to(policy_device)
        
        # Clear cache before policy forward pass (which requires gradients)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        policy_outputs = policy(
            input_ids=mb_input_ids_policy,
            attention_mask=mb_attention_mask_policy,
            output_hidden_states=False,
        )
        policy_probs = F.softmax(policy_outputs.logits, dim=-1)
        del policy_outputs  # Free memory immediately
        # Clear cache after policy forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Move base_topk_indices back to policy_device for gathering
        base_topk_indices_policy = base_topk_indices_cpu.to(policy_device)
        policy_probs_sub = torch.gather(policy_probs, dim=-1, index=base_topk_indices_policy)
        del policy_probs  # Free memory after gathering
        
        # Move policy_probs_sub and base_probs_sub to base_model_device for comparison
        policy_probs_sub_base = policy_probs_sub.to(base_model_device)
        base_probs_sub = base_probs_sub_cpu.to(base_model_device)

        base_probs_sub = base_probs_sub / base_probs_sub.sum(dim=-1, keepdim=True)
        policy_probs_sub_base = policy_probs_sub_base / policy_probs_sub_base.sum(dim=-1, keepdim=True)

        base_log_probs_sub = torch.log(base_probs_sub + 1e-8)
        policy_log_probs_sub = torch.log(policy_probs_sub_base + 1e-8)

        kl_div_topk = F.kl_div(policy_log_probs_sub, base_probs_sub, reduction="none")
        valid_tokens = mb_attention_mask_base.unsqueeze(-1).expand_as(kl_div_topk).float()
        masked_kl_div = kl_div_topk * valid_tokens

        kl_div_per_token = masked_kl_div.sum(dim=-1)
        num_valid_tokens = valid_tokens.sum()
        if num_valid_tokens > 0:
            minibatch_kl_loss = kl_div_per_token.sum() / num_valid_tokens
            total_kl_loss += minibatch_kl_loss
            num_minibatches += 1
        
        # Clean up intermediate tensors
        del mb_input_ids_base, mb_attention_mask_base, mb_input_ids_policy, mb_attention_mask_policy
        del base_topk_indices_policy, base_topk_indices_cpu, base_probs_sub_cpu
        del policy_probs_sub, policy_probs_sub_base, base_probs_sub
        del base_log_probs_sub, policy_log_probs_sub, kl_div_topk, valid_tokens, masked_kl_div
        del kl_div_per_token
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if num_minibatches > 0:
        result = (total_kl_loss / num_minibatches) * kl_coef
        # Return result on policy_device (where it will be used for loss computation)
        return result.to(policy_device)
    return torch.tensor(0.0, device=policy_device, requires_grad=True)


def left_pad_sequence(sequences, batch_first=True, padding_value=0):
    """Left-pad sequences for decoder-only models."""
    if not sequences:
        return torch.empty(0, 0, dtype=torch.long)

    reversed_sequences = [torch.flip(seq, dims=[0]) for seq in sequences]
    padded_reversed = torch.nn.utils.rnn.pad_sequence(
        reversed_sequences,
        batch_first=batch_first,
        padding_value=padding_value,
    )
    return torch.flip(padded_reversed, dims=[1] if batch_first else [0])
