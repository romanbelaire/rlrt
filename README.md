# Auto Red Teaming LLMs

## Overview
This repository contains code for automatically red-teaming large language models (LLMs) using reinforcement learning techniques. The system is designed to find and exploit vulnerabilities in target LLMs by generating adversarial prompts.

## Installation
```bash
git clone https://github.com/romanbelaire/rlrt.git
cd auto-red-teaming-llms
pip install -r requirements.txt
```

## Usage
### Basic Training
```bash
python src/main.py --config_path src/configs/config.json
```

### Evaluation
```bash
python src/test_runner \
  --config-path /tmp/harmbench_eval_config.json \
  --checkpoint-dir $CHECKPOINT_DIR \
  --eval-batch-size 8
```

## Common Flags
- `--data_dir`: Directory containing training/testing data
- `--checkpoint`: Base model checkpoint to use
- `--target_model`: Specific model to use for evaluation (if different from base model)
- `--episodes`: Number of episodes to train/test
- `--adapter_path`: Path to LoRA adapter weights
- `--config_path`: Path to JSON configuration file (see Configuration System below)


You can specify a configuration file when running the main script:

```bash
python main.py --config_path configs/my_agent_config.json
```


## Citation

If you use this code in your research, please cite:

