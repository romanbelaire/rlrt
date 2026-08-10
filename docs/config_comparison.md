# Configuration Comparison: General ARC vs Single-Agent Archer

## Overview

This document compares the configuration files for the original General ARC multiturn experiment and the new Single-Agent Archer experiment with Skywork rewards on the Twenty Questions environment.

## Configuration Files

| File | Purpose | Environment |
|------|---------|-------------|
| `src/configs/general_arc_multiturn.json` | Original dual-agent training | General ARC (AI2) |
| `src/configs/single_agent_archer_twenty_questions.json` | Single-agent archer ablation | Twenty Questions (LMRL-Gym) |

## Key Differences

### Training Configuration

| Parameter | General ARC | Single-Agent Archer | Purpose |
|-----------|-------------|---------------------|---------|
| `hl_only_training` | `false` (implicit) | `false` | Dual vs single agent training |
| `use_skywork_for_token_critic` | N/A | `true` | Enable Skywork reward model |
| `skywork_model_name` | N/A | `"Skywork/Skywork-Reward-V2-Llama-3.1-8B"` | Skywork model identifier |
| `skywork_eval_batch_size` | N/A | `4` | Conservative batch size for rewards |
| `num_masked_marginal_tokens` | N/A | `5` | Number of tokens to mask for marginal rewards |
| `pairwise_marginal_masking` | N/A | `false` | Disable pairwise masking for simplicity |
| `use_task_reward` | N/A | `true` | Enable instruction-following reward |
| `max_marginal_token_positions` | N/A | `10` | Memory limit for token processing |

### Environment Configuration

| Parameter | General ARC | Single-Agent Archer | Purpose |
|-----------|-------------|---------------------|---------|
| `use_mcq_env` | `false` | `true` | Use LMRL-Gym environments |
| `env_task` | `"mc_multiturn"` | `"twenty_questions"` | Environment type |
| `max_new_tokens` | `200` | `100` | Maximum tokens per generation |
| `batch_size` | `64` | `32` | Environment batch size |
| `lmrl_options` | AI2 ARC dataset | Twenty questions settings | Environment-specific options |

### Model Configuration

| Parameter | General ARC | Single-Agent Archer | Purpose |
|-----------|-------------|---------------------|---------|
| `rejection_candidates` | Default | `4` | Number of rejection sampling candidates |
| `use_combined_rejection_scoring` | Default | `true` | Use combined scoring approach |
| `perplexity_start_weight` | Default | `1.0` | Starting weight for perplexity |
| `perplexity_end_weight` | Default | `0.4` | Final weight for perplexity |
| `perplexity_decay_episodes` | Default | `100` | Episodes for perplexity decay |

### GPU Placement Strategy

| Aspect | General ARC | Single-Agent Archer | Purpose |
|--------|-------------|---------------------|---------|
| **GPU Request** | `--gres=gpu:1` | `--gres=gpu:2` | Single vs dual GPU setup |
| **Agent Models** | `cuda:0` | `cuda:0` | Training models placement |
| **Environment Models** | `cuda:0` | `cuda:1` | Inference models placement |
| **Reward Models** | `cuda:0` | `cuda:1` | Skywork and evaluation models |
| **Placement Strategy** | Manual/automatic | Strategic dual-GPU | Optimal resource utilization |

## Detailed Comparison

### Training Parameters

#### General ARC (Original)
```json
{
  "training": {
    "batch_size": 128,
    "minibatch_size": 128,
    "rollout_steps": 1024,
    "episodes": 500,
    "use_per_token_advantages": true,
    "debug_mode": false,
    "enable_model_swapping": false
  }
}
```

#### Single-Agent Archer
```json
{
  "training": {
    "batch_size": 64,           // Reduced for memory efficiency
    "minibatch_size": 32,       // Reduced for Skywork integration
    "rollout_steps": 512,       // Reduced for twenty questions task
    "episodes": 500,            // Same total episodes
    "use_per_token_advantages": true,
    "debug_mode": false,
    "enable_model_swapping": true,  // Enabled for Skywork evaluation
    
    // NEW: Single-agent archer specific
    "hl_only_training": false,
    "use_skywork_for_token_critic": true,
    "skywork_model_name": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "skywork_eval_batch_size": 4,
    
    // NEW: Marginal reward computation
    "num_masked_marginal_tokens": 5,
    "pairwise_marginal_masking": false,
    "max_marginal_token_positions": 10,
    "use_task_reward": true
  }
}
```

### Environment Parameters

#### General ARC (Original)
```json
{
  "environment": {
    "use_mcq_env": false,
    "batch_size": 64,
    "env_task": "mc_multiturn",
    "max_new_tokens": 200,
    "lmrl_options": {
      "dataset_id": "allenai/ai2_arc",
      "subset": "ARC-Challenge",
      "split": "train",
      "shuffle": true,
      "max_examples": 1000,
      "max_turns": 10
    }
  }
}
```

#### Single-Agent Archer
```json
{
  "environment": {
    "use_mcq_env": true,           // Changed: Use LMRL-Gym
    "batch_size": 32,           // Reduced: Memory optimization
    "env_task": "twenty_questions",  // Changed: Twenty questions task
    "max_new_tokens": 100,      // Reduced: Shorter questions
    "lmrl_options": {
      "max_conversation_length": 15,  // Twenty questions specific
      "t5_model_load_path": null,     // Use simple oracle
      "t5_tokenizer_name": "google/flan-t5-xl",
      "t5_use_fp16_activations": false,
      "t5_use_fp16_params": false,
      "t5_max_input_length": 124,
      "t5_max_output_length": 4
    }
  }
}
```

### GPU Placement Configuration

#### General ARC (Original)
```json
// No explicit GPU placement configuration
// Models placed automatically or manually
```

#### Single-Agent Archer
```json
{
  "gpu_placement": {
    "description": "GPU placement strategy for dual-GPU setups",
    "agent_models": {
      "device": "cuda:0",
      "description": "High-level and low-level critics, policies, and actor models"
    },
    "environment_models": {
      "device": "cuda:1",
      "description": "Target model, guard model (if used), and T5 oracle (if enabled)"
    },
    "reward_models": {
      "device": "cuda:1",
      "description": "Skywork reward model and other evaluation models"
    },
    "fallback": {
      "single_gpu": "cuda:0",
      "description": "All models on single GPU if only one available"
    }
  }
}
```

## Memory and Performance Optimizations

### Single-Agent Archer Specific

1. **Reduced Batch Sizes**
   - Training batch: 128 → 64
   - Minibatch: 128 → 32
   - Environment batch: 64 → 32

2. **Skywork Integration**
   - Conservative reward evaluation batch size: 4
   - Model swapping enabled for memory efficiency
   - Chunked forward pass for long sequences

3. **Marginal Reward Computation**
   - Limited token positions: 10 maximum
   - Disabled pairwise masking for speed
   - Efficient token selection strategy

### GPU Placement Benefits

1. **Dual-GPU Strategy**
   - **GPU 0**: Agent models (training and inference)
   - **GPU 1**: Environment and reward models (evaluation)
   - Parallel processing for optimal efficiency

2. **Memory Distribution**
   - Training memory on GPU 0
   - Evaluation memory on GPU 1
   - Reduced memory pressure per device

3. **Scalability**
   - Easy extension to more GPUs
   - Dynamic placement based on model size
   - Load balancing across devices

## Migration Guide

### From General ARC to Single-Agent Archer

1. **Update Training Configuration**
   ```bash
   # Copy and modify the config
   cp src/configs/general_arc_multiturn.json src/configs/single_agent_archer_twenty_questions.json
   ```

2. **Modify Key Parameters**
   - Set `use_mcq_env: true`
   - Change `env_task` to `"twenty_questions"`
   - Add Skywork reward model configuration
   - Adjust batch sizes for memory efficiency

3. **Update SLURM Script**
   ```bash
   # Use the provided script with dual-GPU support
   sbatch train_single_agent_archer_twenty_questions.sh
   ```

4. **GPU Placement Considerations**
   - Ensure 2 GPUs are available
   - Verify CUDA_VISIBLE_DEVICES setting
   - Monitor GPU memory usage

### Environment-Specific Considerations

1. **Twenty Questions Environment**
   - Shorter episodes (15 vs 10 turns)
   - Different reward structure
   - Natural language actions

2. **Skywork Reward Model**
   - Requires HuggingFace access
   - Conservative batch processing
   - Memory-intensive evaluation

3. **Single-Agent Training**
   - Combined critic updates
   - Unified objective function
   - Coordinated strategy-execution

4. **Dual-GPU Setup**
   - Strategic model placement
   - Parallel processing capability
   - Memory-efficient training

## Performance Expectations

### Training Speed
- **Single-Agent Archer**: ~2-3x slower due to Skywork evaluation
- **Memory Usage**: Higher due to dual critics and reward model
- **Convergence**: Similar episode count, different reward dynamics
- **GPU Efficiency**: Better with dual-GPU setup

### Quality Improvements
- **Generation Quality**: Better due to Skywork rewards
- **Strategy Coordination**: Improved due to single agent
- **Task Performance**: Comparable success rates
- **Resource Utilization**: Optimal with strategic GPU placement

## Troubleshooting

### Common Migration Issues

1. **Memory OOM**
   - Reduce `skywork_eval_batch_size` to 2
   - Enable `enable_model_swapping`
   - Reduce `max_marginal_token_positions`

2. **Slow Training**
   - Disable `pairwise_marginal_masking`
   - Reduce `num_masked_marginal_tokens`
   - Increase `chunk_size`

3. **Environment Errors**
   - Verify LMRL-Gym installation
   - Check twenty questions environment availability
   - Validate T5 oracle configuration

4. **GPU Placement Issues**
   - Verify 2 GPUs are available
   - Check CUDA_VISIBLE_DEVICES setting
   - Monitor nvidia-smi output
   - Verify model placement on correct devices

### GPU-Specific Troubleshooting

1. **Dual-GPU Setup**
   - Ensure both GPUs have sufficient memory
   - Monitor memory usage on both devices
   - Verify models are placed correctly
   - Check for GPU communication issues

2. **Single-GPU Fallback**
   - All models will be placed on cuda:0
   - Reduce batch sizes if memory is insufficient
   - Enable aggressive memory optimization
   - Monitor memory usage closely

## Summary

The Single-Agent Archer configuration represents a significant departure from the dual-agent approach, introducing:

- **Unified Architecture**: Single agent with dual critics
- **Quality-Focused Rewards**: Skywork model for generation assessment
- **Efficient Training**: Memory-optimized settings for complex tasks
- **Strategic GPU Placement**: Dual-GPU setup for optimal performance
- **Research Value**: Ablation study for agent architecture comparison

### Key Advantages of New Configuration

1. **Better Resource Utilization**
   - Strategic GPU placement for optimal performance
   - Parallel processing of training and evaluation
   - Memory-efficient model management

2. **Improved Training Efficiency**
   - Single agent reduces coordination overhead
   - Dual-GPU setup enables parallel processing
   - Memory optimization for complex models

3. **Enhanced Research Capabilities**
   - Skywork rewards for quality assessment
   - Twenty questions environment for reasoning tasks
   - Flexible GPU placement for different setups

This configuration enables research into single-agent approaches while maintaining the quality and efficiency of the original system, with the added benefit of strategic GPU placement for optimal performance.
