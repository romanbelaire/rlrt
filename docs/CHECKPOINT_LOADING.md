# Checkpoint Loading Feature

This document explains how to use the new `load_from_checkpoint` feature to continue training from an existing checkpoint.

## Overview

The `load_from_checkpoint` feature allows you to resume training from a previously saved checkpoint. This is useful for:
- Continuing training after interruption
- Fine-tuning models with different hyperparameters
- Resuming experiments with modified configurations

## Configuration

Add the `load_from_checkpoint` field to your training configuration:

```json
{
  "training": {
    "load_from_checkpoint": "path/to/checkpoint/directory",
    "episodes": 1000,
    "lr": 5e-5,
    // ... other training parameters
  }
}
```

## Usage Examples

### 1. Continue Training from Final Model

```bash
python src/main.py --config_path src/configs/checkpoint_loading_example.json
```

This will load the final model from `outputs/previous-run/final_model` and continue training.

### 2. Continue from a Specific Checkpoint

```bash
python src/main.py \
  --config_path src/configs/config.json \
  --load_from_checkpoint outputs/run_20250711_014556/checkpoint_50
```

### 3. Force Continue Training with Missing Models

```bash
python src/main.py \
  --config_path src/configs/config.json \
  --load_from_checkpoint outputs/run_20250711_014556/checkpoint_50 \
  --force
```

This will continue training even if some models (like the high-level policy) are missing from the checkpoint. New models will be created for missing components.

### 4. Command Line Override

You can also specify the checkpoint path via command line:

```bash
python src/main.py \
  --config_path src/configs/config.json \
  --load_from_checkpoint outputs/previous-run/final_model
```

## Checkpoint Directory Structure

The checkpoint directory should contain:

```
checkpoint_directory/
├── config.json              # Optional: saved configuration
├── policy/                  # Required: policy model
│   ├── model/
│   ├── tokenizer/
│   └── meta.json
├── sequence_critic/         # Required: sequence critic
│   └── sequence_value_head.pt
├── token_critic/           # Required: token critic
│   └── token_value_head.pt
├── high_level_policy/      # Optional: high-level policy
│   └── high_level_policy.pt
├── optimizer_states/       # Optional: optimizer states
│   ├── actor_optimizer.pt
│   ├── sequence_critic_optimizer.pt
│   ├── token_critic_optimizer.pt
│   └── high_level_optimizer.pt
└── metrics.json           # Optional: training metrics
```

## Behavior

### Configuration Loading
- If the checkpoint contains a `config.json` file, it will be loaded and used
- If no config is found, the provided configuration will be used
- A warning is printed if config loading fails

### Model Loading
- **Policy**: Loaded from `policy/` directory
- **Sequence Critic**: Loaded from `sequence_critic/` directory  
- **Token Critic**: Loaded from `token_critic/` directory
- **High-Level Policy**: Loaded from `high_level_policy/` directory (if exists)

### Episode Numbering
- **Automatic Detection**: The system automatically detects the episode number from the checkpoint directory name
- **Checkpoint Format**: For directories named `checkpoint_X`, the episode number X is extracted
- **Final Model**: For `final_model` directories, the highest checkpoint number from the parent directory is used
- **Continuation**: Training continues from episode X+1, ensuring no overwriting of existing checkpoints
- **Logging**: All logs and metrics use the actual episode numbers (including offset)

### Optimizer States
- If `optimizer_states/` directory exists, optimizer states will be loaded
- If loading fails, a warning is printed and fresh optimizers are used
- If no optimizer states exist, fresh optimizers are initialized

### Error Handling
- **Missing checkpoint directory**: Raises `FileNotFoundError`
- **Missing model files**: Raises `FileNotFoundError` with details (unless `--force` is used)
- **Model loading failure**: Raises `RuntimeError` and terminates
- **Optimizer loading failure**: Prints warning and continues with fresh optimizers

### Force Flag Behavior
- **Default behavior**: Raises an error if any required models are missing from the checkpoint
- **With `--force`**: Continues training even if some models are missing
  - Missing models will be created fresh
  - Missing optimizer states will be initialized fresh
  - Warning messages will be printed for each missing component
  - Useful for scenarios like missing high-level policy weights where you want to train a new DQN head

## Example Workflow

1. **Initial Training**:
   ```bash
   python src/main.py --config_path src/configs/config.json
   ```
   This creates checkpoints: `checkpoint_0`, `checkpoint_20`, `checkpoint_40`, ..., `final_model`

2. **Continue Training**:
   ```bash
   python src/main.py \
     --config_path src/configs/checkpoint_loading_example.json
   ```
   This loads from `final_model` (episode 100), continues from episode 101, creates: `checkpoint_101`, `checkpoint_121`, etc.

3. **Continue from Specific Checkpoint**:
   ```bash
   python src/main.py \
     --config_path src/configs/config.json \
     --load_from_checkpoint outputs/run_20250711_014556/checkpoint_50
   ```
   This loads from `checkpoint_50`, continues from episode 51, creates: `checkpoint_51`, `checkpoint_71`, etc.

4. **Modify and Continue**:
   ```bash
   # Edit config to change learning rate
   python src/main.py \
     --config_path src/configs/checkpoint_loading_example.json \
     --lr 1e-5
   ```

5. **Continue Training with Missing High-Level Policy**:
   ```bash
   # Continue training even though high-level policy weights are missing
   python src/main.py \
     --config_path src/configs/config.json \
     --load_from_checkpoint outputs/run_20250711_014556/checkpoint_50 \
     --force
   ```
   This will load the policy and critics, create a new high-level policy, and continue training from episode 51.

## Important Notes

- The checkpoint must contain all required model files (policy, sequence critic, token critic) unless `--force` is used
- With `--force`, missing models will be created fresh and training will continue
- Optimizer states are optional but recommended for smooth continuation
- Configuration from the checkpoint takes precedence over provided config
- The feature follows the "fail fast and loud" principle - missing critical files will terminate the program (unless `--force` is used)
- High-level policies are optional and will be skipped if not present

## Troubleshooting

### "Missing model directories in checkpoint"
- Ensure the checkpoint directory contains `policy/`, `sequence_critic/`, and `token_critic/` subdirectories
- Check that the checkpoint was saved correctly

### "Failed to load models from checkpoint"
- Verify that the model files are not corrupted
- Ensure the checkpoint was created with a compatible version of the codebase

### "Warning: No optimizer states found"
- This is normal for checkpoints created before this feature was added
- Training will continue with fresh optimizer states

### "Warning: Failed to load optimizer states"
- The optimizer states may be corrupted or incompatible
- Training will continue with fresh optimizer states

### Using `--force` with missing models
- **"Warning: Policy not found in checkpoint, will create new policy"**: A new policy will be created
- **"Warning: Sequence critic not found in checkpoint, will create new sequence critic"**: A new sequence critic will be created  
- **"Warning: Token critic not found in checkpoint, will create new token critic"**: A new token critic will be created
- **"Warning: High-level policy not found in checkpoint, will create new high-level policy"**: A new high-level policy will be created
- **"Warning: [Optimizer] not found in checkpoint, using fresh optimizer"**: A new optimizer will be initialized

These warnings are expected when using `--force` with incomplete checkpoints and indicate that training will continue with fresh components for the missing models. 