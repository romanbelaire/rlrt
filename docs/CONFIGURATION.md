# Configuration System Documentation

## Overview

This document provides details on the configuration system for the RL agent framework, including important parameters and recent changes.

## Configuration Structure

The configuration system uses a hierarchical structure with several components:

1. **TrainingConfig**: Parameters related to the training process
2. **ModelConfig**: Parameters related to the model architecture
3. **EnvironmentConfig**: Parameters related to the environment and data
4. **AgentConfig**: Master configuration that combines all sub-configurations

## Important Environment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_path` | Path to the dataset file or directory | "HarmBench" |
| `harm_threshold` | Threshold for harm detection | 1.5 |
| `temperature` | Sampling temperature | 0.7 |
| `target_model` | Optional target model to use | None |

## Parameter Consolidation Notes

### `data_path`

The configuration system uses `data_path` to specify the dataset directory or file path for training data.

### Recommended Usage

In configuration files:
```json
"environment": {
  "data_path": "HarmBench/harmbench_behaviors_text_all.csv",
  "harm_threshold": 1.5,
  "temperature": 0.7
}
```

In command line arguments:
```bash
python src/main.py --data_dir HarmBench
```

## Using HarmBench CSV Files

To use the HarmBench CSV files for training:

1. Use the provided configuration file:
   ```bash
   python src/main.py --config_path configs/harmbench_config.json
   ```

2. Or specify the data directory directly:
   ```bash
   python src/main.py --data_dir HarmBench
   ```

3. Or point to a specific CSV file:
   ```bash
   python src/main.py --data_dir HarmBench/harmbench_behaviors_text_all.csv
   ```

The environment will automatically detect CSV files and load them appropriately, using the "Behavior" column for questions when available. 