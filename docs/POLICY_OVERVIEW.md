# Policy Classes Overview

This document provides an overview of all policy classes in the codebase, their purposes, and their save/load capabilities.

## Low-Level Policies (Token Generation)

### 1. `LowLevelPolicy` (`src/policies.py`)
- **Purpose**: Main low-level policy for generating token-level responses
- **Training**: PPO-based training with token-level advantages
- **Save/Load**: ✅ Has `save()` and `@classmethod load()` methods
- **Usage**: Primary policy for generating harmful responses

### 2. `TemplatePolicy` (`src/policies.py`)
- **Purpose**: Template-based low-level policy using predefined response templates
- **Training**: Inherits from `LowLevelPolicy`, uses template matching
- **Save/Load**: ✅ Inherits save/load methods from `LowLevelPolicy`
- **Usage**: Alternative to `LowLevelPolicy` with template-based generation

## High-Level Policies (Strategy Selection)

### 3. `TemplateHighLevelPolicy` (`src/high_level_policy.py`)
- **Purpose**: DQN-based high-level policy that selects templates via Q-learning
- **Policy Kind**: `"dqn"`
- **Training**: DQN with replay buffer and target network
- **Save/Load**: ✅ Has `save()` and `@classmethod load()` methods
- **Usage**: Selects strategies from templates.json for low-level policy guidance

### 4. `PPOTemplateHighLevelPolicy` (`src/ppo_template_policy.py`)
- **Purpose**: PPO-based high-level policy that selects templates via policy gradient methods
- **Policy Kind**: `"ppo"`
- **Training**: PPO with log probabilities and entropy regularization
- **Save/Load**: ✅ Has `save()` and `@classmethod load()` methods
- **Usage**: Selects strategies from templates.json using PPO instead of Q-learning

### 5. `LLMHighLevelPolicy` (`src/llm_high_level_policy.py`)
- **Purpose**: LLM-based high-level policy that generates free-form instructions
- **Policy Kind**: `"llm"`
- **Training**: Inference-only (frozen LLM), optional adapter training
- **Save/Load**: ✅ Has `save()` and `@classmethod load()` methods
- **Usage**: Uses LLM to generate custom instructions for each state

### 6. `HybridHighLevelPolicy` (`src/hybrid_high_level_policy.py`)
- **Purpose**: Combines MLP strategy selection with LLM instruction customization
- **Policy Kind**: `"hybrid"`
- **Training**: MLP component is trainable, LLM component is frozen
- **Save/Load**: ✅ Has `save()` and `@classmethod load()` methods
- **Usage**: Best of both worlds - stable MLP selection + diverse LLM customization

### 7. `FixedHighLevelPolicy` (`src/fixed_high_level_policy.py`)
- **Purpose**: Fixed high-level policy that always returns the same instruction
- **Policy Kind**: `"fixed"`
- **Training**: No training (fixed behavior)
- **Save/Load**: ✅ Has `save()` and `load()` methods (no-op)
- **Usage**: Baseline/comparison policy with deterministic behavior

## Policy Configuration

### Policy Kinds
- `"dqn"`: Template-based selection using DQN Q-learning (default)
- `"ppo"`: Template-based selection using PPO policy gradient
- `"llm"`: Free-form LLM instruction generation
- `"hybrid"`: MLP + LLM combination
- `"fixed"`: Fixed instruction (baseline)

### Training Compatibility
- **DQN Policies**: `TemplateHighLevelPolicy` - uses replay buffer and Q-learning
- **PPO Template Policies**: `PPOTemplateHighLevelPolicy` - uses PPO with log probabilities for template selection
- **LLM Policies**: `LLMHighLevelPolicy` - inference-only, optional adapter training
- **Hybrid Policies**: `HybridHighLevelPolicy` - MLP component uses PPO/DQN, LLM is frozen

## Save/Load Interface

All policies follow a consistent save/load interface:

### Save Method
```python
def save(self, path: str) -> None:
    """Save policy to directory"""
    # Creates directory structure with model weights and metadata
```

### Load Method (Class Method)
```python
@classmethod
def load(cls, path: str, device: Optional[str] = None) -> "PolicyClass":
    """Load policy from directory"""
    # Loads metadata and model weights, returns new instance
```

### Directory Structure
```
policy_directory/
├── meta.json          # Policy metadata (type, kind, config)
├── model.pt           # Model weights
└── [other files]      # Policy-specific additional files
```

## Usage in Training/Evaluation

### Training
- **Agent Trainer**: Uses `high_level_policy.save()` during checkpointing
- **Policy Loading**: Uses `PolicyClass.load()` to restore from checkpoints

### Evaluation
- **Test Runner**: Uses `PolicyClass.load()` to load trained policies
- **Policy Selection**: Based on `policy_type` in meta.json

## Key Features

1. **Hierarchical Architecture**: High-level policies guide low-level token generation
2. **Multiple Training Paradigms**: DQN, PPO, and LLM-based approaches
3. **Template System**: Reusable strategy templates for consistent behavior
4. **Attention Mechanisms**: Token-level attention for strategy selection
5. **Hybrid Approaches**: Combine multiple policy types for better performance
6. **Consistent Interface**: All policies implement the same save/load interface 