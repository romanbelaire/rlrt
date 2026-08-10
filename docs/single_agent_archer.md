# Single-Agent Archer Implementation

## Overview

The Single-Agent Archer is an ablation study that implements a single agent with both high-level and low-level critics, as opposed to the dual-agent approach used in the main MCQ training system.

## Architecture

### Key Components

1. **Single Agent**: One policy that handles both high-level strategy and low-level token generation
2. **High-Level Critic**: `SequenceLevelValueCritic` - Q function that estimates value for state-action pairs
3. **Low-Level Critic**: `TokenLevelValueCriticV` - V function that estimates expected value from state sequences

### Training Objective

The single-agent archer maximizes the objective:

```
J = Σ(A_high - V_low(s[0:i-1]))
```

Where:
- `A_high` = advantage derived from high-level critic (via GAE)
- `V_low(s[0:i-1])` = low-level critic's prediction of expected value from state sequence

## Skywork Reward Model Integration

### Overview

The Single-Agent Archer integrates the **Skywork/Skywork-Reward-V2-Llama-3.1-8B** reward model to provide high-quality generation quality assessment for the token-level critic training. This follows the same pattern as Llamaguard usage in the environment.

### Key Features

1. **Generation Quality Assessment**: Uses Skywork reward model to evaluate response quality instead of just safety
2. **Token-Level Marginal Rewards**: Computes marginal contributions of individual tokens using Skywork scores
3. **Consistent Reward Framework**: Maintains the same marginal reward computation structure as the main system

### Configuration

```python
config = {
    'training': {
        'use_skywork_for_token_critic': True,  # Enable Skywork reward model
        'skywork_model_name': 'Skywork/Skywork-Reward-V2-Llama-3.1-8B',
        'skywork_eval_batch_size': 4,  # Conservative batch size for reward evaluation
        'num_masked_marginal_tokens': 5,  # Number of tokens to mask for marginal rewards
        'pairwise_marginal_masking': False,  # Disable pairwise masking for simplicity
        'use_task_reward': True,  # Enable instruction-following reward
    }
}
```

### Implementation Details

#### Reward Computation

The Skywork reward model is used to compute marginal rewards for token-level critic training:

```python
# Compute marginal rewards using Skywork for better generation quality assessment
marginal_rewards = self.compute_marginal_rewards_with_skywork(
    state=state,
    action=action,
    base_state=base_state,
    environment=environment,
    tokenizer=tokenizer,
    sequence_critic=sequence_critic,
    top_k_tokens=5,
    pairwise_masking=False,
    instructions=instructions,
    skywork_model_name="Skywork/Skywork-Reward-V2-Llama-3.1-8B"
)
```

#### Marginal Reward Calculation

For each token position, the system computes:

1. **J(unmasked)**: Full sequence reward using Skywork
2. **J(masked)**: Masked sequence reward (with specific tokens removed)
3. **Marginal contribution**: J(unmasked) - J(masked)

This provides per-token value estimates that inform the V function training.

#### Memory Optimization

- **Minibatch processing**: Processes responses in small batches to avoid OOM
- **Model swapping**: Moves agent models to CPU during Skywork evaluation
- **Efficient tokenization**: Uses attention-guided token selection to limit computation

### Benefits

1. **Better Generation Quality**: Skywork rewards focus on response quality rather than just safety
2. **Consistent Evaluation**: Same reward model used across all token-level computations
3. **Research Alignment**: Follows the exact usage pattern specified in the Skywork documentation
4. **Memory Efficient**: Optimized for large-scale training with multiple models

### Usage Example

```python
from src.mcq_agent_trainer import SingleAgentArcherTrainer

# Initialize trainer with Skywork reward model
trainer = SingleAgentArcherTrainer(
    actor_optimizer=actor_optimizer,
    sequence_critic_optimizer=seq_optimizer,
    token_critic_optimizer=token_optimizer,
    config=config  # Contains Skywork configuration
)

# Train critics using Skywork rewards
losses = trainer.train_critics_single_agent(
    sequence_critic=sequence_critic,
    token_critic=token_critic,
    state=state,
    action=action,
    base_state=base_state,
    hl_rewards=high_level_rewards,
    ll_rewards=low_level_rewards,
    next_state=next_state,
    environment=environment,
    tokenizer=tokenizer,
    instructions=instructions
)
```

## Training Process

### 1. High-Level Critic Training

The sequence-level critic (Q function) is trained using high-level rewards from the environment:

```python
# Q(s,a) = r_HL + γ * V(s')
q_high_targets = hl_rewards + gamma * next_state_values
sequence_critic_loss = F.mse_loss(sequence_value_current, q_high_targets)
```

### 2. Token-Level Critic Training

The token-level critic (V function) is trained using Skywork marginal rewards:

```python
# Use Skywork rewards for token-level critic training
marginal_rewards = self.compute_marginal_rewards_with_skywork(...)
v_low_targets = marginal_rewards
token_critic_loss = F.mse_loss(token_value_current, v_low_targets)
```

### 3. Policy Training

The policy is updated using the combined objective:

```python
# Maximize J = Σ(A_high - V_low(s[0:i-1]))
policy_loss = -torch.min(surrogate1, surrogate2).mean()
```

## Multi-Objective Weighting

The system supports learnable weights for different objective components:

1. **High-level advantage**: Derived from sequence critic
2. **Low-level value**: Predicted by token critic using Skywork rewards
3. **Instruction following**: Cosine similarity between actions and instructions
4. **Entropy regularization**: Encourages exploration

## Hierarchical Attention

Specialized attention mechanisms are used for different critics:

1. **Sequence Critic**: Full attention over state-action pairs
2. **Token Critic**: State-only attention (V function)
3. **Marginal Computation**: Attention-guided token selection for efficiency

## Configuration Options

### Skywork Reward Model

- `use_skywork_for_token_critic`: Enable/disable Skywork integration
- `skywork_model_name`: HuggingFace model identifier
- `skywork_eval_batch_size`: Batch size for reward evaluation

### Marginal Reward Computation

- `num_masked_marginal_tokens`: Number of tokens to mask
- `pairwise_marginal_masking`: Enable pairwise token masking
- `max_marginal_token_positions`: Memory limit for token processing

### Training Parameters

- `use_task_reward`: Include instruction-following similarity
- `use_per_token_advantages`: Use per-token vs sequence-level advantages
- `naive_decayed_token_reward`: Alternative reward computation method

## Performance Considerations

### Memory Management

- **Model swapping**: Agent models moved to CPU during Skywork evaluation
- **Minibatch processing**: Conservative batch sizes for reward computation
- **Gradient accumulation**: Efficient training with large sequences

### Computational Efficiency

- **Attention-guided selection**: Only process most important tokens
- **Caching**: Reuse Skywork model outputs where possible
- **Parallel processing**: Batch multiple reward computations

## Future Enhancements

1. **Dynamic Reward Weighting**: Learn optimal weights for different reward components
2. **Multi-Model Ensembles**: Combine multiple reward models for robustness
3. **Adaptive Token Selection**: Dynamic selection of tokens for marginal computation
4. **Cross-Modal Rewards**: Integrate visual or audio reward signals

## Troubleshooting

### Common Issues

1. **Memory OOM**: Reduce `skywork_eval_batch_size` or enable model swapping
2. **Slow Training**: Disable pairwise masking or reduce `num_masked_marginal_tokens`
3. **Reward Instability**: Check Skywork model loading and tokenization

### Debug Mode

Enable debug mode for detailed logging:

```python
config['training']['debug_mode'] = True
```

This provides:
- GPU memory usage tracking
- Reward score ranges and statistics
- Token masking progress information
- Model loading status
