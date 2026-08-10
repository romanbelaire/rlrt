# Ablation Studies

This document describes how to set up and run ablation studies to test different components of the hierarchical RL system.

## Ablation 1: Hierarchical vs Flat Agent

This ablation tests the effectiveness of the high-level agent by comparing three model configurations:

1. **Hierarchical Model**: Full hierarchical agent with high-level policy + low-level policy (trained)
2. **Flat Model**: Low-level policy only (trained, no high-level policy)
3. **Baseline Model**: Non-finetuned low-level model (same base model, same system prompts, no training)

## Overview

The ablation study evaluates whether the hierarchical architecture (with high-level policy) provides benefits over a flat architecture (low-level only), and how both compare to a non-finetuned baseline.

## Setup

### 1. Train the Flat Agent

First, train a flat (low-level only) agent using the provided configuration:

```bash
python -m src.main --config_path src/configs/flat_agent_config.json
```

This will train a low-level policy without a high-level policy. The configuration file (`src/configs/flat_agent_config.json`) sets `use_high_level_policy: false`.

**Note**: You should already have a trained hierarchical model checkpoint for comparison. If not, train one using your hierarchical configuration (with `use_high_level_policy: true`).

### 2. Run Ablation Evaluation

Use the ablation evaluator script to compare all three models:

```bash
python -m src.ablation_evaluator \
    --config_path src/configs/flat_agent_config.json \
    --hierarchical_checkpoint outputs/hierarchical-model/checkpoint_150 \
    --flat_checkpoint outputs/flat-agent/checkpoint_150 \
    --baseline_model_name meta-llama/Llama-3.1-8B-Instruct \
    --episodes 100 \
    --max_queries 30 \
    --eval_batch_size 8 \
    --output_dir ablation_results
```

### Arguments

- `--config_path`: Path to configuration JSON file (used for environment and model setup)
- `--hierarchical_checkpoint`: Path to hierarchical model checkpoint directory (optional)
- `--flat_checkpoint`: Path to flat model checkpoint directory (optional)
- `--baseline_model_name`: Base model name for baseline evaluation (optional, uses config if not provided)
- `--episodes`: Number of evaluation episodes per model (default: 100)
- `--max_queries`: Maximum queries per episode (default: 30)
- `--eval_batch_size`: Batch size for evaluation (default: 8)
- `--output_dir`: Directory to save evaluation results (default: "ablation_results")

### 3. Evaluate Individual Models (Alternative)

If you prefer to evaluate models separately, you can use the standard test runner:

#### Hierarchical Model
```bash
python -m src.test_runner \
    --config_path src/configs/general_config.json \
    --checkpoint_dir outputs/hierarchical-model/checkpoint_150 \
    --episodes 100 \
    --max_queries 30
```

#### Flat Model
```bash
python -m src.test_runner \
    --config_path src/configs/flat_agent_config.json \
    --checkpoint_dir outputs/flat-agent/checkpoint_150 \
    --episodes 100 \
    --max_queries 30
```

#### Baseline Model (Non-finetuned)

For the baseline, you need to create a config that matches your training config but without checkpoint loading. The baseline evaluator will build the policy from the base model only.

## Prompt format: Flat vs Frozen high-level

| Ablation              | High-level policy    | Instruction in prompt       | Prompt format                                       |
| --------------------- | -------------------- | --------------------------- | --------------------------------------------------- |
| **Flat (low-level only)** | None                 | None                        | Conversation only (no `[INSTRUCTION]` / `[CONTEXT]` block) |
| **Frozen high-level (fixed)** | FixedHighLevelPolicy | Constant (e.g. "." or "Rephrase this") | Hierarchical: `[INSTRUCTION]` + context + conversation |
| **Frozen high-level (freeform)** | FreeformHighLevelPolicy (frozen) | LLM-generated (not trained) | Hierarchical: `[INSTRUCTION]` + context + conversation |

- **Flat**: No high-level policy; the low-level policy sees only the conversation history. Use `src/configs/flat_agent_config.json` (`use_high_level_policy: false`, `fixed_high_level_instruction: null`).
- **Frozen high-level (fixed)**: A non-trainable high-level policy (`FixedHighLevelPolicy`) supplies a constant instruction every step. Use `src/configs/blankhl-ablation-config.json` (minimal instruction ".") or `src/configs/fixedhl-ablation-config.json` (e.g. "Rephrase this"). When `fixed_high_level_instruction` is set, the builder returns `FixedHighLevelPolicy` even if `use_high_level_policy` is false.
- **Frozen high-level (freeform)**: Use the **freeform** method (LLM generates instructions at rollout) but **skip high-level training** (`freeze_high_level: true`). Rollout uses `FreeformHighLevelPolicy` so the prompt is hierarchical; the high-level policy is not updated. Config: `src/configs/freeform_frozen_hl.json`. Train: `train_freeform_frozen_hl.sh`; eval: `eval_freeform_frozen_hl.sh`.

## Configuration Files

### Flat Agent Config

The `src/configs/flat_agent_config.json` file is configured for training a flat agent:

- `use_high_level_policy: false` - Disables high-level policy
- `fixed_high_level_instruction: null` - No instruction; prompt is conversation only
- Same training parameters as hierarchical config for fair comparison
- Same base model (`hf_checkpoint`) as hierarchical config

### Frozen high-level (ablation) configs

- **blankhl**: `src/configs/blankhl-ablation-config.json` — `fixed_high_level_instruction: "."` (minimal non-whitespace so rollout validation passes). Uses hierarchical format with a constant placeholder instruction.
- **fixedhl**: `src/configs/fixedhl-ablation-config.json` — `fixed_high_level_instruction: "Rephrase this"`. Uses hierarchical format with a fixed textual instruction.
- **freeform frozen HL**: `src/configs/freeform_frozen_hl.json` — `use_freeform_high_level: true`, `freeze_high_level: true`. Freeform high-level policy is used at rollout (LLM-generated instructions, hierarchical format) but not trained; only the low-level policy is trained.

### Matching System Prompts

All three models use the same system prompts for fair comparison:

- The hierarchical model uses system prompts from the high-level policy
- The flat model uses the same system prompts (but without high-level guidance)
- The baseline model uses the same system prompts (from the base model)

## Results Interpretation

The ablation evaluator outputs:

1. **Success rates** for each model configuration
2. **Detailed results** saved to JSON files
3. **Comparison summary** showing relative performance

### Expected Outcomes

- **Hierarchical > Flat**: Indicates the high-level policy provides value
- **Flat > Baseline**: Indicates training improves performance
- **Hierarchical > Baseline**: Confirms overall effectiveness of hierarchical training

## Example Output

```
================================================================================
ABLATION EVALUATION SUMMARY
================================================================================
HIERARCHICAL: 45.00% (45/100)
FLAT: 38.00% (38/100)
BASELINE: 25.00% (25/100)

[ABLATION-EVAL] Results saved to: ablation_results/ablation_results_20240101_120000.json
```

## Notes

- All models should use the same base model checkpoint for fair comparison
- The same evaluation dataset is used for all three models
- System prompts are matched across all configurations
- The baseline model uses the same system prompts but without any training

## Troubleshooting

### Checkpoint Not Found

If a checkpoint path doesn't exist, the evaluator will skip that model and continue with the others. Make sure checkpoint directories contain:
- `policy/` subdirectory with trained policy
- `high_level_policy/` subdirectory (for hierarchical model only)
- `config.json` with training configuration

### Memory Issues

If you encounter GPU memory issues:
- Reduce `eval_batch_size`
- Use quantization for target/guard models
- Evaluate models sequentially instead of in one run

### Configuration Mismatch

Ensure the config file matches the training configuration used for each checkpoint. The evaluator will attempt to load model configuration from checkpoints, but some settings may need to match.

---

## Ablation 2: Adversarial Reward Component (-R(action))

This ablation tests the effect of the adversarial reward component `-R(action)` by comparing models trained with and without it.

### Reward Formulations

The reward function can be configured using `use_rs_minus_ra_reward`:

- **With adversarial reward** (`use_rs_minus_ra_reward: true`): 
  - Reward = R(s') - R(a)
  - Where R(s') = safety score of target's response
  - And R(a) = safety score of agent's action
  - This encourages the agent to generate harmful actions (high R(a)) that elicit safe responses from the target (low R(s'))
  
- **Without adversarial reward** (`use_rs_minus_ra_reward: false`):
  - Reward = R(a)
  - Only considers the safety of the agent's action
  - Does not penalize the agent for generating harmful content

### Setup

#### 1. Train with Adversarial Reward

Train a model with the adversarial reward component enabled:

```bash
python -m src.main --config_path src/configs/flat_agent_config.json
```

This uses `use_rs_minus_ra_reward: true` (default).

#### 2. Train without Adversarial Reward

Train a model with the adversarial reward component disabled:

```bash
python -m src.main --config_path src/configs/flat_agent_no_adversarial_reward_config.json
```

This uses `use_rs_minus_ra_reward: false`.

### Configuration Files

- **With adversarial reward**: `src/configs/flat_agent_config.json` (or any config with `use_rs_minus_ra_reward: true`)
- **Without adversarial reward**: `src/configs/flat_agent_no_adversarial_reward_config.json` (sets `use_rs_minus_ra_reward: false`)

### Evaluation

Compare the two models using the ablation evaluator or test runner:

```bash
# Evaluate model with adversarial reward
python -m src.test_runner \
    --config_path src/configs/flat_agent_config.json \
    --checkpoint_dir outputs/flat-agent/checkpoint_150 \
    --episodes 100 \
    --max_queries 30

# Evaluate model without adversarial reward
python -m src.test_runner \
    --config_path src/configs/flat_agent_no_adversarial_reward_config.json \
    --checkpoint_dir outputs/flat-agent-no-adversarial/checkpoint_150 \
    --episodes 100 \
    --max_queries 30
```

### Expected Differences

- **With adversarial reward (R(s') - R(a))**: 
  - Agent is incentivized to generate harmful actions
  - But also penalized if the target responds harmfully
  - Should produce more sophisticated attacks that avoid triggering harmful responses
  
- **Without adversarial reward (R(a) only)**:
  - Agent is only rewarded for generating harmful actions
  - No penalty for target's response
  - May produce more direct/obvious attacks

### Interpretation

The adversarial reward component `-R(action)` serves to:
1. Encourage harmful agent actions (high R(a))
2. Discourage scenarios where the target responds harmfully (low R(s'))
3. Create a more nuanced reward signal that considers both agent and target behavior

Comparing models with and without this component helps understand its contribution to training effective red teaming agents.

