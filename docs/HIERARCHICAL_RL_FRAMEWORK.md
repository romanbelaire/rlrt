# Hierarchical Reinforcement Learning Framework for Adversarial Red Teaming

## Overview

This framework implements a hierarchical reinforcement learning (HRL) system for adversarial red teaming of language models. The system uses a two-level architecture where a high-level policy selects strategic templates/instructions, and a low-level policy generates concrete responses conditioned on those instructions.

## High-Level Architecture

The framework consists of five main components:

1. **High-Level Policy** - Selects strategic templates/instructions
2. **Low-Level Policy** - Generates responses conditioned on high-level instructions  
3. **High-Level Critic** - Evaluates the value of high-level actions
4. **Low-Level Critic** - Evaluates the value of low-level responses
5. **Environment** - Simulates conversations and provides rewards

### Training Flow

```
1. Environment provides harmful prompts
2. High-level policy selects strategic template/instruction
3. Low-level policy generates response conditioned on instruction
4. Environment evaluates response safety and provides reward
5. Critics estimate value functions for both levels
6. PPO updates both policies using advantages from both critics
```

## Component Details

### 1. High-Level Policy

**Purpose**: Selects strategic templates or generates high-level instructions that guide the low-level policy.

**Implementation**: 
- `TemplateHighLevelPolicy` (DQN-based) in `src/high_level_policy.py`
- `HighLevelTemplatePolicy` (PPO-based) in `src/hierarchical_policies.py`

**Key Methods**:
```python
# DQN-based selection from templates
def select_action(self, input_ids, attention_mask, force_greedy=False) -> int

# PPO-based instruction generation  
def act(self, state, instructions, **kwargs) -> Tuple[List[str], torch.Tensor, torch.Tensor]
```

**Configuration**: Set `use_high_level_policy=True` and `high_level_policy_type` in `ModelConfig`.

### 2. Low-Level Policy

**Purpose**: Generates concrete responses conditioned on high-level instructions.

**Implementation**: `LowLevelConditionedPolicy` in `src/hierarchical_policies.py`

**Key Methods**:
```python
def prepare_conditioned_context(self, original_context, instructions) -> List[str]
def act(self, state, instructions, **kwargs) -> Tuple[List[str], torch.Tensor, torch.Tensor]
```

**Conditioning Process**:
1. Takes original conversation context
2. Prepends high-level instruction: `"Strategy: {instruction}\n\nContext: {context}"`
3. Generates response using base language model

### 3. High-Level Critic

**Purpose**: Estimates the value of high-level actions (template selection/instruction generation).

**Implementation**: `SequenceLevelValueCritic` in `src/critics.py`

**Key Methods**:
```python
def forward(self, input_ids, attention_mask, **kwargs) -> torch.Tensor
def _compute_values(self, hidden_states, attention_mask) -> torch.Tensor
```

**Value Head**: Simple MLP that takes final hidden state and outputs scalar value.

### 4. Low-Level Critic

**Purpose**: Estimates the value of low-level responses at both sequence and token levels.

**Implementations**:
- `SequenceLevelValueCritic` - Estimates overall response value
- `TokenLevelValueCritic` - Estimates per-token values

**Key Methods**:
```python
# Sequence-level value estimation
def forward(self, input_ids, attention_mask, **kwargs) -> torch.Tensor

# Token-level value estimation  
def forward(self, input_ids, attention_mask, **kwargs) -> torch.Tensor
```

### 5. Environment

**Purpose**: Simulates conversations with target models and provides safety-based rewards.

**Implementation**: `MultiStepEnv` in `src/environment.py`

**Key Methods**:
```python
def step(self, actions, **kwargs) -> Tuple[List[List[str]], List[str], torch.Tensor, torch.Tensor]
def score_safety(self, responses, **kwargs) -> torch.Tensor
def reset(self) -> List[List[str]]
```

**Reward Components**:
- Safety score from guard model
- Embedding similarity to harmful prompt
- Perplexity penalty
- Refusal detection

## Training Algorithm

### PPO Training Loop

The main training logic is in `AgentTrainer` class in `src/agent_trainer.py`:

```python
class AgentTrainer:
    def train_on_batch(self, actor, sequence_critic, token_critic, batch_data, 
                      high_level_policy=None, v_high_critic=None, **kwargs):
        # 1. Compute advantages using GAE
        advantages = self.compute_gae(rewards, values, next_values)
        
        # 2. Train critics to better estimate values
        critic_losses = self.train_critics(sequence_critic, token_critic, ...)
        
        # 3. Train high-level policy (if enabled)
        if high_level_policy:
            high_level_losses = self.train_high_level_policy(high_level_policy, ...)
            
        # 4. Train low-level policy
        actor_losses = self.train_actor(actor, ...)
```

### Advantage Computation

**Generalized Advantage Estimation (GAE)** is used for both levels:

```python
def compute_gae(self, rewards, values, next_values, gamma=0.99, gae_lambda=0.95):
    # Standard GAE computation for sequence-level advantages
    # Token-level advantages use marginal reward masking
```

**Marginal Reward Masking**: For token-level advantages, the framework computes marginal rewards by masking individual tokens and computing the difference in value.

### Hierarchical Advantage Computation

The framework computes advantages at both levels:

1. **High-Level Advantages**: Based on the value of template selection/instruction generation
2. **Low-Level Advantages**: Based on the value of conditioned response generation

```python
# High-level advantage computation
high_level_advantages = self.compute_gae(high_level_rewards, v_high_values, next_v_high_values)

# Low-level advantage computation  
low_level_advantages = self.compute_gae(low_level_rewards, v_low_values, next_v_low_values)
```

## Configuration System

The framework uses a comprehensive configuration system in `src/config.py`:

### Main Configuration Classes

```python
@dataclass
class AgentConfig:
    training: TrainingConfig      # PPO hyperparameters
    model: ModelConfig           # Model architecture settings  
    environment: EnvironmentConfig # Environment settings

@dataclass  
class TrainingConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    # ... other PPO parameters

@dataclass
class ModelConfig:
    hf_checkpoint: str = "meta-llama/Llama-3.2-1B-Instruct"
    use_high_level_policy: bool = False
    high_level_policy_type: str = "dqn"  # "dqn", "ppo", "llm", "hybrid"
    # ... other model settings

@dataclass  
class EnvironmentConfig:
    data_path: str = "HarmBench"
    harm_threshold: float = 1.5
    max_steps: int = 5
    # ... other environment settings
```

### Key Configuration Options

**High-Level Policy**:
- `use_high_level_policy`: Enable hierarchical training
- `high_level_policy_type`: Choose between DQN, PPO, LLM, or hybrid
- `templates_path`: Path to strategic templates JSON file

**Training**:
- `alg`: Training algorithm ("ppo")
- `batch_size`: Number of episodes per batch
- `rollout_steps`: Steps per episode
- `use_gae`: Enable Generalized Advantage Estimation

**Environment**:
- `data_path`: Dataset for harmful prompts
- `target_model`: Model to attack
- `guard_model`: Safety evaluation model

## Usage Examples

### Basic Training

```python
from src.main import train_agent
from src.config import AgentConfig

# Load configuration
config = AgentConfig.from_json("config.json")

# Train agent
train_agent(config)
```

### Command Line Training

```bash
python src/main.py --config_path config.json
```

### Hierarchical Training

```bash
python src/main.py --config_path config.json \
    --use_high_level_policy \
    --high_level_policy_type ppo
```

## Key Features

### 1. Template-Based Strategy Selection

The high-level policy can select from predefined strategic templates:

```json
{
  "templates": [
    {"id": "flattery", "instruction": "Use flattery to gain trust"},
    {"id": "authority", "instruction": "Appeal to authority figures"},
    {"id": "urgency", "instruction": "Create sense of urgency"}
  ]
}
```

### 2. Marginal Reward Masking

For token-level training, the framework computes marginal rewards by masking individual tokens:

```python
def compute_marginal_rewards(self, state, action, base_state, environment):
    # Mask each token position and compute value difference
    marginal_rewards = []
    for i in range(action_length):
        masked_action = mask_token_at_position(action, i)
        marginal_reward = value(state, action) - value(state, masked_action)
        marginal_rewards.append(marginal_reward)
    return marginal_rewards
```

### 3. Memory Optimization

The framework includes extensive memory optimization:

- **Device Mapping**: Automatic model placement across GPUs/CPU
- **Gradient Checkpointing**: Reduces memory usage during training
- **Chunked Forward Passes**: Processes large batches in chunks
- **Model Swapping**: Swaps models between CPU/GPU as needed

### 4. Distributed Training

Support for distributed training using Accelerate:

```python
from src.distributed_utils import DistributedSetup

setup = DistributedSetup(config)
model = setup.load_model_with_device_map(model_cls, model_name, device_map)
```

## File Structure

```
src/
├── base.py                    # Abstract interfaces and base classes
├── agent_trainer.py          # PPO training implementation
├── environment.py            # Multi-step conversation environment
├── policies.py               # Low-level policy implementations
├── high_level_policy.py     # DQN-based high-level policy
├── hierarchical_policies.py  # PPO-based hierarchical policies
├── critics.py               # Value function estimators
├── config.py                # Configuration management
├── main.py                  # Training/evaluation entry point
├── utils.py                 # Utility functions
└── distributed_utils.py     # Distributed training utilities
```

## Error Handling

The framework follows a "fail fast and loud" principle for research code:

- No silent error handling
- Detailed error messages with context
- Immediate termination on critical errors
- Comprehensive logging for debugging

This ensures that researchers can quickly identify and fix issues during development.

## Memory Management

The framework includes sophisticated memory management:

1. **Automatic Device Mapping**: Models are automatically placed on optimal devices
2. **Gradient Checkpointing**: Reduces memory usage during backpropagation
3. **Model Swapping**: Models are moved between CPU/GPU as needed
4. **Chunked Processing**: Large batches are processed in smaller chunks
5. **Memory Monitoring**: Real-time memory usage tracking

## Performance Optimizations

1. **Flash Attention**: Uses FlashAttention v2 when available
2. **Mixed Precision**: Supports BF16/FP16 training
3. **Caching**: Compiled kernels are cached for reuse
4. **Batch Processing**: Efficient batch tokenization and processing
5. **Parallel Evaluation**: Safety evaluation runs in parallel

This hierarchical framework provides a comprehensive solution for training adversarial red teaming agents with both strategic planning and tactical execution capabilities. 