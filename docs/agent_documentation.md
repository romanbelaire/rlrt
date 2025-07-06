# Agent Class Documentation

## Overview

The `Agent` class implements a Proximal Policy Optimization (PPO) algorithm for training language models to interact in a Red Team / Blue Team environment. The class is designed to train models to defend against adversarial attacks through reinforcement learning.

## Core Functionality

The Agent combines several key components:

1. **Policy Network (Actor)**: Generates responses to inputs
2. **Value Networks (Critics)**:
   - **Harm Critic**: Evaluates potential harmfulness of responses
   - **Help Critic**: Evaluates helpfulness of responses

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Language Model │     │  Environment    │
│  (Actor)        │◄───►│  (MultiStepEnv) │
│                 │     │                 │
└────────┬────────┘     └────────▲────────┘
         │                       │
         │                       │
         │                       │
┌────────▼────────┐             │
│                 │             │
│  Value Critics  │             │
│  (Harm/Help)    │─────────────┘
│                 │
└─────────────────┘
```

## Key Methods

### Initialization

```python
def __init__(self, policy_type, data_path, hf_checkpoint, auth_token=None, 
             hyperparameters=None, use_finetuned=False, use_lora=False, ...)
```

Initializes the PPO agent with configurable components:
- Policy type (model architecture)
- Data source
- Base model checkpoint
- LoRA adapters configuration
- Model singletons for efficient memory usage
- Custom harm critics

### Learning

```python
def learn(self)
```

The main training loop that:
1. Collects experiences via rollouts
2. Computes advantages and returns
3. Updates the policy and value networks using PPO
4. Tracks and logs training metrics

### Rollout Collection

```python
def rollout(self, num_steps)
```

Collects interactions with the environment by:
1. Starting or continuing conversations
2. Getting agent responses
3. Receiving environment rewards
4. Tracking conversation states
5. Processing data for batch training

### PPO Training

```python
def _train_on_batch(self, batch_data, num_epochs=None)
```

Implements the PPO algorithm by:
1. Interleaving value and policy updates
2. Computing advantages using GAE
3. Calculating PPO clipped surrogate objective
4. Applying policy gradient updates

### Advantage Calculation

```python
def _compute_advantages(self, rewards, values, next_values=None, gamma=0.99, 
                        gae_lambda=0.95, dones=None, token_level=False, td_error=None)
```

Computes advantages for policy learning using Generalized Advantage Estimation (GAE).

## Special Features

### LoRA Adapter Management

The agent supports efficient fine-tuning using Low-Rank Adaptation (LoRA):

```python
def initialize_adapter_models(self)
def switch_lora_adapter(self, adapter_type)
def save_lora_adapter(self, adapter_type)
def load_lora_adapter(self, adapter_type)
```

Different adapter types can be used for different conversation stages:
- `attention_holistic`: General conversation
- `lm_head_rephrase`/`attention_rephrase`: Response rephrasing
- `lm_head_followup`/`attention_followup`: Follow-up handling

### Question Breakdown Strategy

```python
def breakdown_rollout(self, t)
```

An alternative rollout strategy that breaks complex harmful questions into atomic components.

### Memory Management

```python
def to_cuda(self)
def to_cpu(self)
```

Methods to efficiently manage GPU memory by moving models between devices.

## Reward Structure

The agent uses multiple reward signals:

1. **Harm Rewards**: Penalize harmful responses
2. **Help Rewards**: Encourage helpful responses
3. **Compliance Rewards**: Measure how well the target model complies with requests
4. **Action Help Rewards**: Assess the helpfulness of the agent's actions

## Hyperparameters

Key configurable parameters include:

- Learning rates (`lr`, `value_lr`)
- Discount factor (`gamma`)
- PPO clipping parameter (`clip`)
- Batch and minibatch sizes
- LoRA configuration (`lora_r`, `lora_alpha`, `lora_dropout`)
- Loss coefficients (`ce_coef`, `adv_coef`, `help_weight`)

## Usage

The Agent is typically instantiated with a policy type, data path, and model checkpoint:

```python
agent = Agent(
    policy_type="Transformer",
    data_path="path/to/data",
    hf_checkpoint="meta-llama/Llama-3.2-1B-Instruct",
    hyperparameters={
        "model_name": "my-ppo-model",
        "lr": 1e-6,
        "batch_size": 64,
        "epochs": 3
    }
)

# Start training
agent.learn()
```

## Dependencies

- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Custom environments and models

## Evaluation

The agent regularly logs performance metrics including:
- Average episode length
- Average rewards (harm, help)
- Loss values (actor, critics)
- Semantic similarity scores

## Model Saving and Loading

Models and adapters are saved periodically during training and can be loaded for continued training or inference. 