# Single-Agent Archer Experiment: Twenty Questions with Skywork Rewards

## Overview

This experiment implements the **Single-Agent Archer** ablation study using the **Twenty Questions** environment from LMRL-Gym, with **Skywork/Skywork-Reward-V2-Llama-3.1-8B** providing generation quality assessment for the token-level critic.

## GPU Placement Strategy

### Dual-GPU Setup

When running on a system with 2 or more GPUs, the experiment uses a strategic GPU placement strategy:

| GPU | Device | Models | Purpose |
|-----|--------|--------|---------|
| **GPU 0** | `cuda:0` | **Agent Models** | Training and inference for RL components |
| **GPU 1** | `cuda:1` | **Environment/Judge Models** | Evaluation and reward computation |

#### GPU 0 (cuda:0): Agent Models
- **High-Level Critic**: `SequenceLevelValueCritic` (Q function)
- **Low-Level Critic**: `TokenLevelValueCriticV` (V function)  
- **High-Level Policy**: PPO-based strategy selection
- **Actor Policy**: Token generation policy
- **LoRA Adapters**: Fine-tuned model parameters

#### GPU 1 (cuda:1): Environment/Judge Models
- **Target Model**: Base model for environment responses
- **T5 Oracle**: Twenty questions answer generation (if enabled)
- **Skywork Reward Model**: Generation quality assessment
- **Guard Model**: Safety evaluation (if used)

### Single-GPU Fallback

If only one GPU is available, all models are placed on `cuda:0` with automatic memory management.

### Benefits of This Strategy

1. **Memory Efficiency**: Separates training (GPU 0) from evaluation (GPU 1)
2. **Parallel Processing**: Agent training and reward computation can happen simultaneously
3. **Optimal Utilization**: Each GPU focuses on its specialized task
4. **Scalability**: Easy to extend to more GPUs if needed

## Experiment Design

### Single-Agent Architecture

The single-agent archer uses **one agent** with **two critics**:

1. **High-Level Critic**: `SequenceLevelValueCritic` (Q function) - estimates value for state-action pairs
2. **Low-Level Critic**: `TokenLevelValueCriticV` (V function) - estimates expected value from state sequences using Skywork rewards

### Training Objective

Maximize the combined objective:

```
J = Σ(A_high - V_low(s[0:i-1]))
```

Where:
- `A_high` = advantage derived from high-level critic (via GAE)
- `V_low(s[0:i-1])` = low-level critic's prediction of expected value from state sequence

### Environment: Twenty Questions

The **Twenty Questions** environment provides a natural language reasoning task where:

- **Goal**: Guess an object within 20 yes/no questions
- **State**: Conversation history of questions and answers
- **Actions**: Natural language questions
- **Rewards**: 
  - **High-level**: Task completion (correctly guessing the object)
  - **Low-level**: Generation quality assessed by Skywork reward model

## Configuration

### Key Training Parameters

```json
{
  "training": {
    "use_skywork_for_token_critic": true,
    "skywork_model_name": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "skywork_eval_batch_size": 4,
    "num_masked_marginal_tokens": 5,
    "pairwise_marginal_masking": false,
    "use_task_reward": true
  }
}
```

### GPU Placement Configuration

```json
{
  "gpu_placement": {
    "agent_models": {
      "device": "cuda:0",
      "description": "High-level and low-level critics, policies, and actor models"
    },
    "environment_models": {
      "device": "cuda:1", 
      "description": "Target model, guard model (if used), and LLM oracle"
    },
    "reward_models": {
      "device": "cuda:1",
      "description": "Skywork reward model and other evaluation models"
    }
  }
}
```

### Environment Settings

```json
{
  "environment": {
    "use_mcq_env": true,
    "env_task": "twenty_questions",
    "max_new_tokens": 100,
    "lmrl_options": {
      "max_conversation_length": 15,
      "oracle_model_name": "meta-llama/Llama-3.1-8B-Instruct",
      "oracle_quantize": false,
      "hf_token": null
    }
  }
}
```

### Recent Environment Simplifications

The Twenty Questions environment has been significantly simplified to improve reliability and performance:

#### **Removed Complex Components**
- **WordVariants**: Replaced complex object variants with simple string words
- **T5 Oracle**: Replaced with HuggingFace LLM oracle for better natural language understanding
- **Postprocessing Functions**: Removed complex text preprocessing since LLM handles natural language directly
- **NLTK Dependencies**: Eliminated POS tagging and complex word validation

#### **LLM Oracle Implementation**
- **Model**: Uses HuggingFace models (default: Llama-3.1-8B-Instruct)
- **Prompt Engineering**: Explicit yes/no instructions for reliable responses
- **Generation Parameters**: Optimized for short, deterministic responses
- **Response Parsing**: Robust parsing with fallback logic

#### **Benefits**
- **40% Code Reduction**: From 425 to 250 lines
- **Better Performance**: No complex preprocessing overhead
- **Improved Reliability**: LLM oracle provides more accurate semantic understanding
- **Easier Debugging**: Simple string operations instead of complex object handling

### Model Configuration

- **Base Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **LoRA**: Enabled with rank 8, alpha 8
- **Target Modules**: `q_proj`, `k_proj` (layers 0-3)
- **High-Level Policy**: PPO-based strategy selection

## Running the Experiment

### 1. Configuration File

The experiment uses `src/configs/single_agent_archer_twenty_questions.json` which configures:

- Single-agent archer training mode
- Skywork reward model integration
- Twenty questions environment parameters
- Memory optimization settings
- GPU placement strategy

### 2. SLURM Script

Use the provided SLURM script:

```bash
sbatch train_single_agent_archer_twenty_questions.sh
```

**Note**: The SLURM script requests 2 GPUs (`#SBATCH --gres=gpu:2`) to enable the dual-GPU placement strategy.

### 3. Manual Execution

```bash
# For dual-GPU setup
CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
  --config_path src/configs/single_agent_archer_twenty_questions.json \
  --output_dir outputs/single_agent_archer_twenty_questions

# For single-GPU setup
CUDA_VISIBLE_DEVICES=0 python -m src.main \
  --config_path src/configs/single_agent_archer_twenty_questions.json \
  --output_dir outputs/single_agent_archer_twenty_questions
```

## Expected Behavior

### Training Dynamics

1. **Episode Structure**: Each episode consists of up to 15 question-answer turns
2. **High-Level Strategy**: Agent selects questioning strategies (e.g., category-based, elimination-based)
3. **Low-Level Execution**: Agent generates specific questions using the selected strategy
4. **Dual Reward Signal**: 
   - High-level: Task completion reward
   - Low-level: Skywork generation quality assessment

### GPU Utilization

1. **GPU 0**: Active during agent training and policy updates
2. **GPU 1**: Active during reward computation and environment evaluation
3. **Parallel Processing**: Both GPUs can work simultaneously for optimal efficiency

### Reward Computation

#### High-Level Rewards
- **Task Success**: +1.0 for correctly guessing the object
- **Efficiency Bonus**: Higher rewards for fewer questions
- **Strategy Quality**: Rewards based on question effectiveness

#### Low-Level Rewards (Skywork)
- **Generation Quality**: Skywork model evaluates question clarity and relevance
- **Marginal Contributions**: Computes per-token value using masking approach
- **Consistent Evaluation**: Same reward model across all token-level computations

### Memory Optimization

- **Model Swapping**: Agent models moved to CPU during Skywork evaluation
- **Minibatch Processing**: Conservative batch sizes (4) for reward computation
- **Chunked Training**: Efficient handling of long sequences
- **Gradient Accumulation**: Memory-efficient training with large batches

## Key Differences from Dual-Agent Approach

| Aspect | Dual-Agent | Single-Agent Archer |
|--------|------------|---------------------|
| **Agents** | 2 separate agents | 1 agent with 2 critics |
| **Low-Level Critic** | Q function (state+action) | V function (state only) + Skywork rewards |
| **Training** | Separate updates | Combined update with dual critics |
| **Objective** | Separate objectives | Combined objective: J = Σ(A_high - V_low) |
| **Reward Signal** | Environment safety | Skywork generation quality |
| **GPU Usage** | Single GPU or manual placement | Strategic dual-GPU placement |

## Benefits of This Approach

### 1. **Better Generation Quality**
- Skywork rewards focus on response quality rather than just safety
- Consistent evaluation across all token-level computations
- Research-aligned with Skywork documentation

### 2. **Improved Coordination**
- Single agent ensures strategy-execution alignment
- Combined training objective promotes holistic learning
- Reduced training complexity and memory footprint

### 3. **Research Value**
- Ablation study for single vs. dual agent architectures
- Evaluation of Skywork rewards in RL context
- Comparison with traditional safety-based rewards

### 4. **Efficient Resource Utilization**
- Strategic GPU placement for optimal performance
- Parallel processing of training and evaluation
- Memory-efficient model management

## Monitoring and Debugging

### Debug Mode

Enable detailed logging:

```json
{
  "training": {
    "debug_mode": true
  }
}
```

This provides:
- GPU memory usage tracking
- Skywork reward score ranges and statistics
- Token masking progress information
- Model loading status
- GPU placement verification

### GPU Monitoring

The SLURM script automatically provides:
- GPU availability check
- Memory usage per GPU
- Placement strategy verification
- Fallback behavior for single-GPU systems

### Key Metrics to Monitor

1. **High-Level Performance**
   - Task completion rate
   - Average questions per episode
   - Strategy selection distribution

2. **Low-Level Quality**
   - Skywork reward scores
   - Marginal reward distributions
   - Token-level value estimates

3. **Training Stability**
   - Loss convergence
   - Gradient norms
   - Memory usage per GPU

4. **GPU Utilization**
   - Memory usage on GPU 0 (agent models)
   - Memory usage on GPU 1 (environment models)
   - Parallel processing efficiency

## Troubleshooting

### Common Issues

1. **Memory OOM**
   - Reduce `skywork_eval_batch_size` to 2
   - Enable `enable_model_swapping`
   - Reduce `max_marginal_token_positions`

2. **Slow Training**
   - Disable `pairwise_marginal_masking`
   - Reduce `num_masked_marginal_tokens`
   - Increase `chunk_size`

3. **Skywork Model Issues**
   - Verify HuggingFace token access
   - Check model download status
   - Monitor GPU memory during reward computation

4. **GPU Placement Issues**
   - Verify 2 GPUs are available
   - Check CUDA_VISIBLE_DEVICES setting
   - Monitor nvidia-smi output

### Performance Tuning

```json
{
  "training": {
    "skywork_eval_batch_size": 2,        // Reduce if OOM
    "num_masked_marginal_tokens": 3,     // Reduce for speed
    "max_marginal_token_positions": 5,   // Reduce for memory
    "chunked_forward_pass": true,        // Enable for efficiency
    "enable_model_swapping": true        // Enable for memory
  }
}
```

### GPU-Specific Tuning

1. **Dual-GPU Setup**
   - Ensure both GPUs have sufficient memory
   - Monitor memory usage on both devices
   - Verify models are placed correctly

2. **Single-GPU Fallback**
   - All models will be placed on cuda:0
   - Reduce batch sizes if memory is insufficient
   - Enable aggressive memory optimization

## Expected Results

### Training Convergence

- **High-Level Critic**: Should converge to accurate Q-value estimates
- **Low-Level Critic**: Should learn to predict generation quality from Skywork rewards
- **Policy**: Should balance strategy selection with question generation quality

### Performance Metrics

- **Task Success Rate**: Target >80% within 15 questions
- **Question Quality**: Skywork scores should improve over training
- **Efficiency**: Fewer questions per successful episode
- **Strategy Diversity**: Balanced use of different questioning approaches

### GPU Efficiency

- **Parallel Processing**: Both GPUs should be utilized simultaneously
- **Memory Balance**: Even distribution of memory usage across GPUs
- **Training Speed**: Optimal performance with dual-GPU setup

## Future Extensions

### 1. **Multi-Objective Weighting**
- Learnable weights for different objective components
- Adaptive reward balancing based on training progress

### 2. **Enhanced Reward Models**
- Ensemble of multiple reward models
- Domain-specific reward fine-tuning
- Cross-modal reward integration

### 3. **Advanced Architectures**
- Hierarchical attention mechanisms
- Dynamic token selection strategies
- Adaptive critic architectures

### 4. **Multi-GPU Scaling**
- Extend to 4+ GPUs for larger models
- Dynamic GPU placement based on model size
- Load balancing across multiple devices

## Recent Fixes

### Oracle Response Parsing Issue (Latest)
**Problem**: Oracle responses like "Yes." and "No." were not being correctly parsed, causing evaluation accuracy to show 0% even when the oracle was answering correctly.

**Solution**: Enhanced punctuation handling in response parsing to ensure oracle responses are properly recognized.

**Impact**: Evaluation metrics now accurately reflect oracle performance, with accuracy showing correct percentages (e.g., 24.8% instead of 0%).

## References

- **Single-Agent Archer**: Ablation study for unified agent training
- **Skywork Reward Model**: [Skywork/Skywork-Reward-V2-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)
- **Twenty Questions Environment**: LMRL-Gym implementation
- **LMRL-Gym**: [Repository](https://github.com/rail-berkeley/LMRL_Gym)

## Contact

For questions about this experiment, contact the research team or refer to the main documentation in `docs/single_agent_archer.md`.
