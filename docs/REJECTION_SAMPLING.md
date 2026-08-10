# Combined Rejection Sampling

This document describes the combined rejection sampling feature that uses both `v_high` and perplexity scores to select the best candidate responses during rollout collection.

## Overview

The rejection sampling process has been enhanced to use a combination of two scoring criteria:

1. **v_high score**: Measures the expected reward using the high-level critic (Q_high)
2. **Perplexity score**: Measures the naturalness/fluency of the generated text

Both scores are batch normalized before combination to ensure fair comparison across different scales.

## Configuration

The combined rejection sampling can be configured through the following parameters in the model configuration:

```json
{
  "model": {
    "use_combined_rejection_scoring": true,
    "use_perplexity_only": false,
    "perplexity_start_weight": 0.5,
    "perplexity_end_weight": 0.1,
    "perplexity_decay_episodes": 100,
    "rejection_candidates": 8
  }
}
```

### Parameters

- `use_combined_rejection_scoring` (bool): Whether to use combined scoring (default: true)
- `use_perplexity_only` (bool): Whether to use only perplexity for rejection sampling (overrides use_combined_rejection_scoring) (default: false)
- `perplexity_start_weight` (float): Starting weight for perplexity score (default: 0.5)
- `perplexity_end_weight` (float): Final weight for perplexity score after annealing (default: 0.1)
- `perplexity_decay_episodes` (int): Number of episodes to decay perplexity weight (default: 100)
- `rejection_candidates` (int): Number of candidates to generate per state (default: 8)

## How It Works

### 1. Candidate Generation
For each state, N candidate responses are generated using the low-level policy.

### 2. Scoring
Each candidate is scored using both criteria:

#### v_high Score
- Computed using the high-level critic (Q_high)
- Measures expected reward: R(s') - R(a)
- Higher scores indicate better expected outcomes

#### Perplexity Score
- Computed using the target model (or a frozen Llama model for OpenAI targets)
- Measures text naturalness/fluency
- Lower perplexity indicates more natural text
- **Note**: The perplexity score is negated and normalized so that higher is better

### 3. Batch Normalization
Both scores are batch normalized to ensure fair comparison:

```python
# For v_high (higher is better)
normalized_v_high = (v_high_scores - v_high_mean) / v_high_std

# For perplexity (lower is better, so we negate)
normalized_perplexity = -(perplexity_scores - perplexity_mean) / perplexity_std
```

### 4. Weight Annealing
The weights are annealed over training episodes:

```python
progress = min(current_episode / perplexity_decay_episodes, 1.0)
perplexity_weight = perplexity_start_weight + (perplexity_end_weight - perplexity_start_weight) * progress
v_high_weight = 1.0 - perplexity_weight  # v_high gets the remaining weight
```

This ensures that:
- Early in training: Higher perplexity weight helps ensure natural text generation
- Later in training: Higher v_high weight focuses on reward optimization

### 5. Combination
The final score is computed as:

```python
combined_score = v_high_weight * normalized_v_high + perplexity_weight * normalized_perplexity
```

### 5. Selection
The candidate with the highest combined score is selected (with epsilon-greedy exploration).

## Usage

### Enable Combined Scoring with Annealing
To use combined rejection sampling with weight annealing, set in your configuration:

```json
{
  "model": {
    "use_combined_rejection_scoring": true,
    "perplexity_start_weight": 0.5,
    "perplexity_end_weight": 0.1,
    "perplexity_decay_episodes": 100
  }
}
```

### Disable Combined Scoring
To fall back to the original v_high-only scoring:

```json
{
  "model": {
    "use_combined_rejection_scoring": false
  }
}
```

### Use Perplexity Only
To use only perplexity for rejection sampling (using the target model):

```json
{
  "model": {
    "use_perplexity_only": true
  }
}
```

**Note**: When `use_perplexity_only` is enabled, it overrides `use_combined_rejection_scoring` and uses only the target model's perplexity computation for candidate selection.

### Adjust Annealing Schedule
You can customize the annealing schedule:

```json
{
  "model": {
    "use_combined_rejection_scoring": true,
    "perplexity_start_weight": 0.6,  // Higher initial perplexity weight
    "perplexity_end_weight": 0.05,   // Lower final perplexity weight
    "perplexity_decay_episodes": 200 // Slower decay over more episodes
  }
}
```

## Example Configuration

See `src/configs/combined_rejection_scoring_config.json` for a complete example configuration.

## Debug Output

When debug mode is enabled, the rejection sampling process will output:

```
REJECTION SAMPLING DEBUG (Episode 0):
  Low-level epsilon: 0.100
  Number of candidates per state: 8
  Total candidates generated: 8
  Scoring method: Combined v_high + perplexity (batch normalized, annealed)
  Annealing: perplexity 0.5 → 0.1 over 100 episodes
  Current weights (episode 0): v_high=0.50, perplexity=0.50
  Candidate scores for first state: [0.234, -0.156, 0.789, ...]
  Selected candidate index: 2
  Selection method: highest-scoring
```

## Implementation Details

### Function: `compute_combined_rejection_score`

```python
def compute_combined_rejection_score(
    self, 
    state: torch.Tensor, 
    action: torch.Tensor, 
    base_state: torch.Tensor, 
    environment: Any,
    tokenizer: Any,
    current_episode: int = 0,
    perplexity_start_weight: float = 0.5,
    perplexity_end_weight: float = 0.1,
    perplexity_decay_episodes: int = 100
) -> torch.Tensor:
```

This function:
1. Computes annealed weights based on current episode
2. Computes v_high scores using the existing critic
3. Computes perplexity scores using the environment's perplexity model
4. Batch normalizes both scores
5. Combines them with the annealed weights
6. Returns the combined scores

### Fallback Behavior

If the perplexity model is not available (e.g., in `MultiStepEnv`), the function will:
- Return neutral perplexity scores (all 1.0)
- Show a warning message (only once per training run)
- Continue with the combination using these neutral scores
- This ensures the system works even without perplexity computation

## Benefits

1. **Better Text Quality**: Perplexity scoring helps select more natural, fluent responses
2. **Balanced Selection**: Combines reward optimization with text quality
3. **Adaptive Training**: Annealing ensures appropriate focus at different training stages
4. **Configurable**: Annealing schedule can be adjusted based on your training needs
5. **Backward Compatible**: Can be disabled to use original v_high-only scoring
6. **Robust**: Handles edge cases like zero standard deviation gracefully

### Annealing Benefits

The annealing approach provides several advantages:

- **Early Training**: Higher perplexity weight ensures the model learns to generate natural, fluent text
- **Mid Training**: Balanced weights allow the model to optimize both text quality and reward
- **Late Training**: Higher v_high weight focuses on reward optimization as the model becomes more competent
- **Smooth Transition**: Gradual weight changes prevent sudden shifts in behavior

## Trade-offs

- **Computational Cost**: Perplexity computation adds overhead
- **Memory Usage**: Requires additional model for perplexity computation
- **Complexity**: More complex scoring mechanism than single-criterion approaches

## Troubleshooting

### Perplexity Model Not Available
If you see warnings about perplexity model initialization failing:
- The system will fall back to neutral perplexity scores
- Combined scoring will still work but with reduced perplexity influence
- Check that the Llama model for perplexity computation is accessible

### Zero Standard Deviation
If all candidates have identical scores:
- The system handles this gracefully by using mean-centered scores
- No division by zero errors occur

### Memory Issues
If you encounter memory issues:
- Reduce `rejection_candidates` to generate fewer candidates
- Use smaller batch sizes
- Consider disabling combined scoring if perplexity computation is too expensive 