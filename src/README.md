# RL Agent Framework Source Code

This directory contains the refactored source code for the reinforcement learning agent framework.

## Directory Structure

- `base.py`: Abstract base classes and interfaces for the agent architecture
- `config.py`: Configuration management classes using dataclasses
- `critics.py`: Implementations of critic networks for value estimation
- `environment.py`: Environment implementations for agent interaction
- `policies.py`: Policy implementations for action generation
- `agent_trainer.py`: PPO implementation for training agents using actor-critic architecture
- `distributed_utils.py`: Utilities for memory management and device mapping using Accelerate
- `utils.py`: Utility functions for memory management, device handling, and performance optimization
- `main.py`: Main entry point for training and evaluation
- `trainer_usage_example.py`: Example code demonstrating how to use the AgentTrainer
- `optimization_example.py`: Example script demonstrating memory and performance optimizations
- `configs/`: Directory containing configuration file templates
  - `sample_config.json`: Example configuration file
  - `accelerate_config_sample.yaml`: Example Accelerate configuration

## Documentation

- `agent_documentation.md`: Original agent documentation
- `refactoring_plan.md`: Plan for the refactoring process
- `refactoring_progress.md`: Progress report on the refactoring

## Usage

To train an agent using a configuration file:

```bash
python src/main.py --config_path src/configs/sample_config.json
```

To test an agent using the test runner:

```bash
python -m src.test_runner --config-path src/configs/sample_config.json --checkpoint-dir path/to/checkpoint
```

You can override specific configuration values via command line arguments:

```bash
python src/main.py --config_path src/configs/sample_config.json --learning_rate 5e-5 --batch_size 16
```

## Memory Management with Accelerate

The framework uses Accelerate primarily for its auto device mapping functionality, which is essential for working with large language models under memory constraints. This allows models to be automatically partitioned across available GPU(s) and CPU memory in an optimal way.

### Using Accelerate Device Mapping

```bash
python src/main.py --use_accelerate --config_path src/configs/sample_config.json
```

### Command-line Arguments for Memory Management

- `--use_accelerate`: Enable Accelerate for memory-optimized model loading
- `--mixed_precision`: Choose mixed precision mode (fp16, bf16, or no) to reduce memory requirements
- `--num_gpus`: Number of GPUs to use for model partitioning (up to 2)

## Memory-Optimized Model Loading

The `distributed_utils.py` module includes specialized functions for loading models with optimized memory usage:

```python
from distributed_utils import DistributedSetup, calculate_optimal_device_map

# Calculate an optimal device map for a model
device_map = calculate_optimal_device_map(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    max_gpu_memory="10GiB",  # Limit per GPU
    num_gpus=2               # Use up to 2 GPUs
)

# Set up Accelerate
setup = DistributedSetup(config)

# Load model with device map
model = setup.load_model_with_device_map(
    model_cls=AutoModelForCausalLM,
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"},
    device_map="auto"  # Or pass custom device map
)
```

## Training Architecture

The framework uses a modular training architecture:

1. **Components**: Policy, Critics, and Environment are separate components that can be instantiated independently
2. **AgentTrainer**: Implements the PPO algorithm for training the policy and critic networks
3. **Configuration**: All hyperparameters are managed through a structured configuration system
4. **Memory Management**: Optimizes model memory usage across available devices

The training loop follows these steps:
1. Initialize policy, critics, and environment based on configuration
2. Set up memory-optimized loading if using Accelerate
3. Collect rollouts from the environment using the current policy
4. Train critics to better estimate value functions
5. Compute advantages using Generalized Advantage Estimation (GAE)
6. Train policy with clipped objective function
7. Repeat the process for the specified number of episodes

## Memory Efficiency

For training with very large models, the framework provides several memory optimization options:

1. **Auto Device Mapping**: Automatically distributes model layers across available hardware
2. **Gradient Checkpointing**: Reduces memory usage by recomputing intermediate activations
3. **Mixed Precision Training**: Uses lower precision (FP16/BF16) to reduce memory footprint
4. **CPU Offloading**: Offloads optimizer states and parameters to CPU when not needed
5. **Gradient Accumulation**: Splits batches into micro-batches to reduce peak memory usage
6. **Dynamic Batch Sizing**: Automatically adjusts batch size based on available GPU memory
7. **Tensor Caching**: Avoids redundant tensor operations by caching results

### Utility Functions

The framework includes a comprehensive set of utility functions in `utils.py` for memory and performance optimization:

```python
# Memory management
clear_gpu_memory()  # Clear GPU memory and run garbage collection
print_memory_stats("Label")  # Print current memory usage

# Device utilities
device = get_device()  # Get best available device
tensor = to_device(tensor, device)  # Move tensor to device efficiently
batch = batch_to_device(batch_dict, device)  # Move all tensors in a batch to device

# Performance optimization
with optimize_memory_usage():
    # Operations with optimized memory usage
    
with tensor_operation(tensor, device, requires_grad=True) as t:
    # Operations with the tensor, handles device and gradient tracking

# Caching
result = cached_computation("key_prefix", compute_fn, *args, **kwargs)
optimal_batch = dynamic_batch_size(tensors, target_memory_usage=0.8)
output = registered_forward_pass(model, input_tensor)
```

## Implementation Status

This is a refactored version of the original agent implementation. The main improvements include:

1. Better organization with separate modules for different components
2. Clear interfaces between components through base classes
3. Comprehensive configuration management system
4. Type hints throughout the codebase
5. Improved readability and maintainability
6. Complete PPO implementation in the AgentTrainer module
7. Proper separation of concerns between agent components and training logic
8. Distributed training support with Accelerate

## Code Optimization Recommendations

Based on regular code reviews, here are current optimization recommendations for the codebase:

### Memory Management

1. **Optimize GPU Memory Usage**
   - Implement consistent device management across all modules
   - Use `to_appropriate_devices()` method consistently instead of ad-hoc device transfers
   - Add automatic memory cleanup after large operations

2. **Reduce Redundant Tensor Allocations**
   - Cache intermediate tensors when possible
   - Use in-place operations where appropriate
   - Implement a tensor cache for frequently reused tensors

3. **Efficient Model Loading**
   - Complete the implementation of `BaseModelSingleton.to_appropriate_devices()`
   - Implement model unloading when switching between components

### Performance Optimizations

1. **Batch Processing Efficiency**
   - Optimize batch creation in `AgentTrainer.train_on_batch()`
   - Implement dynamic batch sizing based on available memory

2. **Reduce Redundant Forward Passes**
   - Cache model outputs when appropriate
   - Implement a forward pass registry to avoid duplicated computations

3. **Optimize Distributed Training**
   - Implement asynchronous gradient updates
   - Add more efficient tensor gathering mechanisms

### Code Structure Improvements

1. **Standardize Device Management**
   - Create a centralized device manager class
   - Implement consistent device transfer patterns
   
2. **Refactor Duplicate Code**
   - Extract common tokenization patterns into utility functions
   - Create shared utility methods for tensor manipulation

3. **Streamline Configuration Management**
   - Implement configuration validation at loading time
   - Add configuration presets for common scenarios

### Implementation Priorities

0. **Always**
   - fail fast and loudly
   - always raise errors

1. **High Priority**
   - Complete the unimplemented methods in `BaseModelSingleton`
   - Implement consistent device management
   - Address redundant tensor allocations

2. **Medium Priority**
   - Optimize batch processing
   - Streamline distributed training

3. **Lower Priority**
   - Refactor duplicate code
   - Improve configuration management 

## Template Policy and Branch-Aware GAE

The codebase includes a specialized `TemplatePolicy` that generates multiple template options for each input state. This approach is designed for sample efficiency in reinforcement learning by learning from all possible template branches, not just the ones that were selected by the environment.

### Key Features

- **Multiple Template Generation**: Generates multiple possible responses for each input state using templates
- **Branch Tracking**: Maintains a tree structure of branches that tracks which branches were followed vs. not followed
- **Non-Standard GAE Calculation**: Implements specialized Generalized Advantage Estimation (GAE) for unchosen branches
- **Sample Efficiency**: Learns from all template branches, leading to much higher sample efficiency

### Template Policy

The `TemplatePolicy` extends `TransformerPolicy` to add template handling:

```python
from src.policies.low_level import TemplatePolicy

policy = TemplatePolicy(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    use_lora=True,
    templates_path="path/to/templates.json",
    debug_mode=True
)

# Generate multiple template responses
template_responses, template_tokens, template_log_probs = policy.act(state)
```

### Branch-Aware GAE Calculation

The `AgentTrainer` class includes specialized GAE calculation for branch-aware training:

```python
# Initialize trainer with branch-aware GAE capability
trainer = AgentTrainer(
    actor_optimizer=optimizer,
    harm_critic_optimizer=critic_optimizer,
    clip_ratio=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01
)

# Collect data with branch tracking
rollout_data = policy.rollout(
    env=env,
    num_steps=100,
    max_batch_size=4
)

# Train with branch-aware GAE
metrics = trainer.train_on_batch(
    actor=policy,
    harm_critic=harm_critic,
    batch_data=rollout_data
)
```

### How Branch-Aware GAE Works

1. **For Chosen Branches**: Standard GAE calculation is used, propagating values through the actual trajectory.
2. **For Unchosen Branches**: A special Q-learning style advantage is calculated using the reward and critic value.

This approach ensures that both chosen and unchosen branches contribute to policy learning, maximizing sample efficiency.

### Running the Example

To run the template policy example:

```
python src/template_policy_example.py --checkpoint "meta-llama/Llama-3.2-1B-Instruct" --debug
```

This will demonstrate template generation, branch tracking, and branch-aware GAE calculation in action.

## Error Handling

The codebase follows strict error handling principles to ensure robustness in a research environment:

### Core Principles

1. **Fail Fast and Loudly**: Errors are never silently caught and ignored. All exceptions propagate upward to be properly handled or to terminate the program.

2. **Detailed Error Information**: All exceptions include detailed context information, traceback, and suggestions for resolution when possible.

3. **Centralized Error Management**: The `error_handling.py` module provides consistent error handling functions that are used throughout the codebase.

### Error Types

The framework defines several custom exception types for different categories of errors:

- `RLAgentError`: Base exception class for all framework errors
- `ConfigurationError`: For errors in configuration setup
- `ModelError`: For errors related to model operations
- `TrainingError`: For errors during the training process
- `EnvironmentError`: For errors in environment interactions
- `DistributedError`: For errors in distributed training

### Usage Example

```python
from error_handling import raise_error, ModelError, log_and_raise

# Example 1: Raising a typed error with context
try:
    # Some risky operation
    model.to_device("invalid_device")
except Exception as e:
    raise_error(ModelError, "Failed to move model to device", original_error=e)

# Example 2: Logging and re-raising an exception
try:
    # Another risky operation
    result = compute_complex_operation()
except Exception as e:
    log_and_raise(e, context="During complex computation")
```

### No Silent Failures

Unlike many libraries that catch exceptions to prevent crashes, this framework intentionally allows exceptions to propagate. This approach ensures that:

1. Bugs are discovered early in the development cycle
2. Error states are not masked by silent failures
3. Researchers have complete visibility into what's happening in their code

This strict error handling approach is crucial for research code, where understanding exactly how and why a model fails is as important as watching it succeed. 