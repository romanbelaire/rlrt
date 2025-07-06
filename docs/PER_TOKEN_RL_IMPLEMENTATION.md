# Per-Token Reinforcement Learning Implementation

This document describes the implementation of true per-token RL where each token generation is treated as an individual timestep/action in the RL framework.

## Overview

### Previous Approach (Sequence-Level)
- **Marginal Rewards**: Computed per-token [batch_size, action_len]
- **Value Function**: Aggregated to sequence-level [batch_size] 
- **Advantages**: Computed at sequence-level, then broadcast to all tokens
- **Result**: All tokens in a sequence get the same advantage signal

### New Approach (Per-Token)
- **Marginal Rewards**: Computed per-token [batch_size, action_len]
- **Value Function**: Maintained per-token [batch_size, action_len] using rewards-to-go
- **Advantages**: Computed per-token using GAE [batch_size, action_len]
- **Result**: Each token gets its own advantage based on its individual contribution

## Key Changes

### 1. Value Function (`v_low`)

**New Implementation:**
```python
def v_low(self, marginal_rewards: torch.Tensor) -> torch.Tensor:
    """Returns per-token values [batch_size, action_len]"""
    batch_size, action_len = marginal_rewards.shape
    values = torch.zeros_like(marginal_rewards)
    
    # Compute rewards-to-go for each token position
    for t in range(action_len):
        future_rewards = marginal_rewards[:, t:]
        discount_factors = self.gamma ** torch.arange(action_len - t)
        values[:, t] = (future_rewards * discount_factors.unsqueeze(0)).sum(dim=1)
    
    return values
```

**Deprecated Implementation:**
```python
def v_low_deprecated(self, marginal_rewards: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """Returns sequence-level values [batch_size]"""
    # Recursive aggregation to single value per sequence
```

### 2. GAE Computation (`compute_gae_per_token`)

**New Implementation:**
```python
def compute_gae_per_token(self, marginal_rewards, values, attention_mask, ...):
    """
    Computes per-token advantages using GAE.
    
    Each token is treated as an individual timestep:
    - Token t gets advantage based on: r_t + γ*V(t+1) - V(t)
    - GAE accumulation happens per token position
    
    Returns: [batch_size, action_len] advantages
    """
```

### 3. Task Reward Distribution

**New Approach:**
- Task rewards (instruction-following similarity) are distributed equally across valid tokens
- Each token gets `sim_score / num_valid_tokens` additional reward
- Maintains per-token granularity

**Old Approach:**
- Task rewards added as sequence-level bonus
- All tokens get the same boost

## Configuration

### Enable/Disable Per-Token RL

```json
{
  "training": {
    "use_per_token_advantages": true,  // true = per-token, false = sequence-level
    ...
  }
}
```

### For Ablation Studies

**Experiment 1: Per-Token RL (Default)**
```json
"use_per_token_advantages": true
```

**Experiment 2: Sequence-Level RL (Baseline)**
```json
"use_per_token_advantages": false
```

## Implementation Details

### Attention Masking
- Per-token GAE respects padding tokens via attention masks
- Only valid (non-padding) tokens participate in advantage computation
- Advantages are normalized only across valid tokens

### Memory Efficiency
- Per-token values require more memory: [batch_size, action_len] vs [batch_size]
- Compatible with existing chunking mechanisms
- GAE computation is batched for efficiency

### Backward Compatibility
- The main `compute_gae()` method automatically detects tensor dimensions
- 2D tensors → per-token GAE
- 1D tensors → sequence-level GAE (deprecated)

## Theoretical Justification

### Why Per-Token RL Makes Sense

1. **Natural Alignment**: Each token generation is indeed an action in language modeling
2. **Fine-Grained Credit Assignment**: Tokens get advantages based on their individual contribution
3. **Better Learning Signal**: Harmful tokens get negative advantages, helpful tokens get positive advantages
4. **Marginal Reward Compatibility**: Directly uses the per-token marginal rewards computed by the system

### Mathematical Framework

For token position t in sequence:

**Value Function:**
```
V(s_t) = Σ_{k=t}^{T} γ^{k-t} * r_k
```

**Advantage Function:**
```
A(s_t, a_t) = r_t + γ*V(s_{t+1}) - V(s_t)
```

**GAE:**
```
A^{GAE}_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
where δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
```

## Usage Examples

### Training with Per-Token RL
```bash
# Default: uses per-token advantages
python -m src.main --config src/configs/config.json
```

### Ablation: Sequence-Level RL
```bash
# Modify config.json to set "use_per_token_advantages": false
python -m src.main --config src/configs/config.json
```

### Monitoring
Look for log messages indicating the approach being used:
```
# Per-token approach will show:
Using chunked actor training: batch_size=128, seq_len=407, chunk_size=8
[shape checks show advantages as [batch_size, action_len]]

# Sequence-level approach will show:
[shape checks show advantages as [batch_size] then broadcast]
```

## Expected Results

### Per-Token RL Benefits
- **Better Credit Assignment**: Harmful tokens get penalized, helpful tokens rewarded
- **Faster Convergence**: More precise learning signals
- **Improved Safety**: Direct optimization of token-level harmfulness

### Performance Considerations
- **Memory**: ~2-4x more memory for advantages (depends on sequence length)
- **Computation**: Slightly more expensive GAE computation
- **Chunking**: Still compatible with memory optimization frameworks

## Debugging

### Shape Assertions
The code includes shape assertions to catch dimension mismatches:
```python
assert advantages.shape == ratio.shape, f"Advantage shape {advantages.shape} must match ratio shape {ratio.shape}"
```

### Common Issues
1. **Memory OOM**: Per-token advantages use more memory - reduce chunk sizes if needed
2. **Shape Mismatches**: Ensure attention masks are correctly computed
3. **NaN Advantages**: Check that valid tokens exist in sequences

## Future Extensions

### Possible Improvements
1. **Hierarchical Per-Token**: Combine with high-level policy for multi-scale token optimization
2. **Attention-Weighted GAE**: Use attention weights to bias GAE computation
3. **Dynamic Masking**: Mask tokens based on relevance scores rather than just padding

This implementation provides a solid foundation for true per-token RL while maintaining backward compatibility for ablation studies. 