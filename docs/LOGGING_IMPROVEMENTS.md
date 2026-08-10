# Metrics Logging and Checkpoint Improvements

## Overview

This document describes the improvements made to the metrics logging and checkpoint saving system to ensure that each training run has its own isolated directories and files.

## Problem Statement

### Previous Issues:
1. **Metrics Logging**: All episodes from different runs were appended to the same CSV files in the output directory
2. **Checkpoint Overwriting**: Checkpoints from new runs would overwrite checkpoints from previous runs
3. **No Run Isolation**: No way to distinguish between different training sessions

### Example of Previous Behavior:
```
outputs/normal-agent-8b8t-taskreward/
├── episodic_rewards.csv          # Mixed data from multiple runs
├── losses_per_episode.csv        # Mixed data from multiple runs
├── input_action_pairs.csv        # Mixed data from multiple runs
├── checkpoint_1/                 # Overwritten by each new run
├── checkpoint_2/                 # Overwritten by each new run
└── final_model/                  # Overwritten by each new run
```

## Solution Implemented

### 1. Run-Specific Directory Structure

Each training run now creates unique directories based on timestamp:

```
outputs/normal-agent-8b8t-taskreward/
├── run_20250711_014556/          # Run-specific output directory
│   ├── checkpoint_1/
│   ├── checkpoint_2/
│   └── final_model/
└── run_20250711_020000/          # Another run
    ├── checkpoint_1/
    └── final_model/

logs/
├── run_20250711_014556/          # Run-specific log directory
│   ├── episodic_rewards.csv      # Only episodes from this run
│   ├── losses_per_episode.csv    # Only losses from this run
│   └── input_action_pairs.csv    # Only pairs from this run
└── run_20250711_020000/          # Another run's logs
    ├── episodic_rewards.csv
    ├── losses_per_episode.csv
    └── input_action_pairs.csv
```

### 2. Unique Run ID Generation

Each training session generates a unique run ID:
```python
run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# Example: "run_20250711_014556"
```

### 3. Experiment ID Enhancement

Experiment IDs now include the run ID for better traceability:
```python
experiment_id = f"{run_id}_template_{num_templates}"
# Example: "run_20250711_014556_template_14"
```

## Code Changes

### 1. AgentTrainer.train() Method

**File**: `src/agent_trainer.py`

**Changes**:
- Added run ID generation at the start of training
- Created run-specific output and log directories
- Updated all CSV file paths to use run-specific log directory
- Updated checkpoint saving to use run-specific output directory
- Modified return value to include run_id

**Key Changes**:
```python
# Generate unique run ID for this training session
import datetime
run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create run-specific directories
run_output_dir = os.path.join(output_dir, run_id)
run_log_dir = os.path.join("logs", run_id)

os.makedirs(run_output_dir, exist_ok=True)
os.makedirs(run_log_dir, exist_ok=True)

# Updated CSV paths
rewards_csv_path = os.path.join(run_log_dir, "episodic_rewards.csv")
losses_csv_path = os.path.join(run_log_dir, "losses_per_episode.csv")
pairs_csv_path = os.path.join(run_log_dir, "input_action_pairs.csv")

# Updated checkpoint paths
save_checkpoint_fn(..., output_dir=run_output_dir, ...)

# Return run_id for plotting
return policy, sequence_critic, token_critic, episode_rewards, run_id
```

### 2. Main Training Function

**File**: `src/main.py`

**Changes**:
- Updated to handle the additional run_id return value
- Modified plotting to use run-specific log directory

**Key Changes**:
```python
# Updated function call
policy, sequence_critic, token_critic, episode_rewards, run_id = trainer.train(...)

# Updated plotting
run_log_dir = os.path.join("logs", run_id)
losses_csv_path = os.path.join(run_log_dir, "losses_per_episode.csv")
generate_training_plots(losses_csv_path, run_log_dir)
```

## Benefits

### 1. **No More Data Mixing**
- Each run's metrics are stored in separate CSV files
- No risk of mixing episodes from different training sessions
- Clear separation of experimental results

### 2. **No More Checkpoint Overwriting**
- Each run has its own checkpoint directory
- Previous runs' checkpoints are preserved
- Easy to compare models from different training sessions

### 3. **Better Experiment Tracking**
- Unique run IDs make it easy to identify specific training sessions
- Experiment IDs include run context for better traceability
- Clear directory structure for organizing results

### 4. **Improved Debugging**
- Easy to locate logs and checkpoints for specific runs
- No confusion about which data belongs to which run
- Better reproducibility of experiments

## Usage

### Running Training
The changes are transparent to the user. Training runs exactly as before:

```bash
python src/main.py --config src/configs/config.json
```

### Finding Results
After training, results are organized by run:

```bash
# Find the latest run
ls -la outputs/normal-agent-8b8t-taskreward/ | tail -5

# Check logs for a specific run
ls -la logs/run_20250711_014556/

# View metrics for a specific run
head -10 logs/run_20250711_014556/episodic_rewards.csv
```

### Comparing Runs
Easy to compare different training sessions:

```bash
# Compare final rewards across runs
for run in logs/run_*; do
    echo "=== $run ==="
    tail -1 $run/episodic_rewards.csv
done
```

## Backward Compatibility

The changes maintain backward compatibility:
- Existing CSV files in output directories are preserved
- Existing checkpoints are not affected
- Training scripts continue to work without modification
- Only new runs use the improved directory structure

## Future Enhancements

Potential future improvements:
1. **Run Metadata**: Store run configuration and hyperparameters in each run directory
2. **Run Summaries**: Generate summary reports for each run
3. **Run Comparison Tools**: Utilities to compare metrics across runs
4. **Automatic Cleanup**: Options to clean up old runs after a certain time
5. **Run Naming**: Allow custom run names instead of just timestamps 