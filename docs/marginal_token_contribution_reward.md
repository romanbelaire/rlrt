# Marginal Token Contribution Reward

## Overview

The marginal token contribution reward measures how much each individual token in an agent's action contributes to the safety or quality of the target's response. This per-token reward signal enables fine-grained training of token-level critics and provides interpretable attribution of which tokens drive harmful or helpful responses.

## Core Formula

For each token position `i`, the marginal contribution is computed as:

```
M(τ_i) = J(unmasked) - J(masked_i)
```

Where:
- **J(unmasked)** = R(s'_unmasked) - R(a_unmasked)
- **J(masked_i)** = R(s'_masked_i) - R(a_masked_i)
- **R(s')** = Safety/quality score of the target's response to the agent's action
- **R(a)** = Safety/quality score of the agent's action itself

The difference `R(s') - R(a)` captures the **differential reward**: how much the target's response improves (or degrades) safety compared to the agent's action alone.

## Mathematical Intuition

The marginal reward answers: *"If we remove token `i`, how does the overall safety/quality change?"*

- **Positive marginal reward**: Token `i` increases safety/quality (removing it makes things worse)
- **Negative marginal reward**: Token `i` decreases safety/quality (removing it makes things better)
- **Zero marginal reward**: Token `i` has no effect

## Implementation Details

### Token Selection

To avoid computing marginal rewards for all tokens (computationally expensive), the system uses **attention-guided token selection**:

1. Extract last hidden states from the sequence critic
2. Compute token importance as L2 norm of hidden states
3. Select top-k most important tokens (default: 5)
4. Only compute marginal rewards for selected tokens

This focuses computation on tokens the critic deems most relevant to harmfulness/quality.

### Masking Strategies

**Single-Token Masking**: For each selected token position `i`, create a masked version where token `i` is removed (via attention mask), then compute J(masked_i).

**Pairwise Masking** (optional): For pairs of tokens `(i, j)`, create masked versions with both tokens removed. The marginal contribution is split equally between the two tokens: `(J(unmasked) - J(masked_{i,j})) / 2`.

### Reward Sources

The system supports two reward models:

1. **LlamaGuard**: Safety-focused reward model for red-teaming scenarios
   - `R(s')` = Safety score of target response
   - `R(a)` = Safety score of agent action

2. **Skywork Reward Model**: Quality-focused reward model for generation tasks
   - `R(s')` = Generation quality score of target response
   - `R(a)` = Generation quality score of agent action

### Additional Components

**Instruction-Following Reward**: Optionally adds cosine similarity between actions and instructions, distributed equally across all valid tokens:
```
marginal_rewards[token_i] += similarity_score / num_valid_tokens
```

## Computational Efficiency

The implementation includes several optimizations:

- **Chunked Processing**: Processes masked versions in minibatches to avoid OOM errors
- **Model Swapping**: Moves agent models to CPU during reward model evaluation
- **Memory Management**: Explicit tensor cleanup and GPU cache clearing
- **Selective Computation**: Only processes top-k tokens instead of all tokens

## Usage in Training

Marginal rewards are used to train **token-level critics** (V or Q functions):

```python
# Compute marginal rewards
marginal_rewards = compute_marginal_rewards(
    state=state,
    action=action,
    base_state=base_state,
    environment=environment,
    tokenizer=tokenizer,
    sequence_critic=sequence_critic,
    top_k_tokens=5,
    pairwise_masking=False
)

# Train token critic using marginal rewards as targets
token_critic_loss = F.mse_loss(
    token_critic_values,  # Predicted per-token values
    marginal_rewards      # Ground-truth marginal contributions
)
```

## Key Configuration Parameters

- `num_masked_marginal_tokens`: Number of top-k tokens to process (default: 5)
- `pairwise_marginal_masking`: Enable pairwise masking (default: False)
- `max_marginal_token_positions`: Memory limit for token processing (default: 10)
- `use_task_reward`: Include instruction-following reward (default: True)
- `Q_as_reward`: Use sequence critic Q-values instead of environment rewards (default: False)

## Advantages

1. **Fine-Grained Attribution**: Identifies which specific tokens drive harmful/helpful responses
2. **Interpretability**: Provides per-token explanations for model behavior
3. **Efficient Training**: Enables token-level value function learning
4. **Flexible Reward Models**: Supports both safety and quality-focused rewards
5. **Scalable**: Attention-guided selection limits computational cost

## Limitations

- Only processes top-k tokens, missing contributions from less-attended tokens
- Pairwise masking can be computationally expensive
- Requires multiple forward passes through reward models
- Sensitive to reward model quality and calibration

