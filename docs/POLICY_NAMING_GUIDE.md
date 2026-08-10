# Policy Naming System Guide

This document explains the reconciled naming system for policies in the codebase after fixing naming inconsistencies.

## Overview

The codebase has two main policy types:
1. **Main/Low-level Policy**: Controls token generation (the agent being trained)
2. **High-level Policy**: Controls instruction/strategy generation (hierarchical RL component)

## Main Policy Configuration

### Algorithm Selection (`training.alg`)
- **Location**: `config.training.alg`
- **Default**: `"ppo"`
- **Valid Values**: `["ppo", "reinforce", "ppo-pretrain"]`
- **Purpose**: Controls the training algorithm for the main agent
- **Current Status**: Only validated, not actively used (defaults to PPO-style training)

## High-Level Policy Configuration

### High-Level Policy Type (`model.high_level_policy_type`)
- **Location**: `config.model.high_level_policy_type`
- **Default**: `"dqn"`
- **Valid Values**: `["dqn", "ppo", "llm", "hybrid"]`
- **Purpose**: Controls the high-level policy architecture when hierarchical RL is enabled

### Mapping Table

| Config Value | Implementation Class | `policy_kind` | Training Method | Description |
|-------------|---------------------|---------------|-----------------|-------------|
| `"dqn"` | `TemplateHighLevelPolicy` | `"dqn"` | DQN Q-learning | Uses Q-learning with replay buffer (default) |
| `"ppo"` | `PPOTemplateHighLevelPolicy` | `"ppo"` | PPO policy gradient | Uses PPO for template selection |
| `"llm"` | `LLMHighLevelPolicy` | `"llm"` | No training (frozen) | Uses frozen LLM for instruction generation |
| `"hybrid"` | `HybridHighLevelPolicy` | `"hybrid"` | Hybrid training | MLP selects strategy, LLM customizes |

## Implementation Details

### Class Hierarchy

```
High-Level Policies:
├── HighLevelTemplatePolicy (hierarchical_policies.py)
│   ├── policy_kind: configurable ("dqn" or "ppo")
│   ├── Used for: Template-based instruction generation
│   └── Training: DQN or PPO based on policy_kind
├── TemplateHighLevelPolicy (high_level_policy.py)
│   ├── policy_kind: "dqn" (fixed)
│   ├── Used for: Pure DQN-based template selection
│   └── Training: DQN with replay buffer
├── LLMHighLevelPolicy (llm_high_level_policy.py)
│   ├── policy_kind: "llm" (fixed)
│   ├── Used for: Free-form instruction generation
│   └── Training: None (frozen model)
└── HybridHighLevelPolicy (hybrid_high_level_policy.py)
    ├── policy_kind: "hybrid" (fixed)
    ├── Used for: Combined MLP + LLM approach
    └── Training: MLP component trained, LLM component frozen
```

### Training Logic

The `AgentTrainer` checks the `policy_kind` attribute to determine training method:

```python
policy_kind = getattr(high_level_policy, "policy_kind", None)
if policy_kind == "dqn":
    # Use DQN training with replay buffer
    high_level_policy.optimise_q_head()
elif policy_kind == "ppo":
    # Use PPO training with rollout data
    self.train_high_level_policy(...)
elif policy_kind == "llm":
    # Skip training (frozen model)
    pass
elif policy_kind == "hybrid":
    # Train MLP component based on its policy_kind
    mlp_policy = high_level_policy.mlp_policy
    # ... train MLP component
```

## Configuration Examples

### DQN-based High-Level Policy
```json
{
  "model": {
    "use_high_level_policy": true,
    "high_level_policy_type": "dqn"
  }
}
```

### PPO-based High-Level Policy
```json
{
  "model": {
    "use_high_level_policy": true,
    "high_level_policy_type": "ppo"
  }
}
```

### Hybrid High-Level Policy
```json
{
  "model": {
    "use_high_level_policy": true,
    "high_level_policy_type": "hybrid",
    "hybrid_policy_enabled": true
  }
}
```

### No High-Level Policy (Flat Agent)
```json
{
  "model": {
    "use_high_level_policy": false
  }
}
```

## Backward Compatibility

- **"mlp"** alias has been removed for clarity
- All configurations should use **"dqn"** for DQN-style training
- Backward compatibility is maintained through automatic conversion in the policy creation logic

## Key Changes Made

1. **Standardized `policy_kind` attributes**: All classes now set consistent `policy_kind` values
2. **Added configuration validation**: Invalid high-level policy types are caught at startup
3. **Implemented proper mapping**: Config values correctly map to implementation classes
4. **Added hybrid policy training**: Complete training logic for hybrid policies
5. **Removed confusing alias**: "mlp" alias removed for clarity
6. **Improved documentation**: Clear separation between config and implementation naming

## Error Handling

The system now provides clear error messages for invalid configurations:

```python
# Invalid high-level policy type
ValueError: Invalid high-level policy type: invalid_type. Must be one of ['dqn', 'ppo', 'llm', 'hybrid']

# Invalid policy_kind in implementation
ValueError: Invalid policy_kind: invalid_kind. Must be one of ['dqn', 'ppo']
```

## Best Practices

1. **Use "dqn" for DQN-style training** in all configurations
2. **Check `policy_kind` attribute** when implementing new training logic
3. **Validate configurations early** to catch naming issues at startup
4. **Document any new policy types** in this guide
5. **Maintain backward compatibility** when making changes

## Future Considerations

- All policy types now use clear, descriptive names
- Easy to add more specific policy types as needed (e.g., "dqn-double", "ppo-clip")
- Clear separation between configuration and implementation naming 