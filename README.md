# Auto Red Teaming LLMs

## Overview
This repository contains code for automatically red-teaming large language models (LLMs) using reinforcement learning techniques. The system is designed to find and exploit vulnerabilities in target LLMs by generating adversarial prompts.

## Features
- Reinforcement learning-based red teaming of LLMs
- Support for different target models
- Token-wise reward attribution
- Customizable attack strategies
- Flexible configuration management system

## Installation
```bash
git clone https://github.com/yourusername/auto-red-teaming-llms.git
cd auto-red-teaming-llms
pip install -r requirements.txt
```

## Usage
### Basic Training
```bash
python main.py --train --data_dir data/questions --checkpoint meta-llama/Llama-3.2-1B-Instruct
```

### Evaluation
```bash
python main.py --test --test_model your_model_name --data_dir data/questions --eval_dataset PKU-Alignment/PKU-SafeRLHF
```

### Using a Separate Target Model
For consistent evaluation or to test against a specific model version:
```bash
python main.py --test --data_dir data/questions --target_model meta-llama/Llama-3.2-3B-Instruct
```

### Fine-tuning with LoRA
```bash
python main.py --train --data_dir data/questions --use_finetuned --adapter_path ./models/fine-tuned/llama-lora
```

### Detuning Model Safety Alignment
```bash
python main.py --train --data_dir data/questions --detune
```

## Key Parameters
- `--train`: Enable training mode
- `--test`: Enable testing mode
- `--data_dir`: Directory containing training/testing data
- `--checkpoint`: Base model checkpoint to use
- `--target_model`: Specific model to use for evaluation (if different from base model)
- `--episodes`: Number of episodes to train/test
- `--help_weight`: Weight for balancing harm vs help advantages (0-1)
- `--attack_type`: Type of attack to perform (rephrase or breakdown)
- `--use_finetuned`: Use a fine-tuned model with LoRA adapters
- `--adapter_path`: Path to LoRA adapter weights
- `--detune`: Enable detuning of model safety alignment
- `--config_path`: Path to JSON configuration file (see Configuration System below)

# Configuration System

The codebase now includes a structured configuration management system for organizing hyperparameters and settings.

## Configuration Structure

The system consists of several configuration classes:

1. **TrainingConfig**: Parameters related to the training process
   - Learning rate, batch size, optimizer settings, etc.

2. **ModelConfig**: Parameters related to the model architecture
   - Model paths, LoRA settings, model dimensions, etc.

3. **EnvironmentConfig**: Parameters related to the training environment
   - Dataset paths, reward settings, environment constraints, etc.

4. **AgentConfig**: Master configuration that aggregates all others
   - Contains training, model, and environment configs
   - Adds agent-specific parameters (name, devices, etc.)

## Using the Configuration System

### Creating Configurations

You can create configuration objects directly by specifying parameters:

```python
from config import TrainingConfig, ModelConfig, EnvironmentConfig, AgentConfig

# Create configuration objects
training_config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=8
)

model_config = ModelConfig(
    base_model_path="EleutherAI/pythia-1.4b",
    use_lora=True
)

env_config = EnvironmentConfig(
    dataset_path="data/conversations.jsonl",
    max_conversation_length=512
)

# Create master agent configuration
agent_config = AgentConfig(
    training=training_config,
    model=model_config,
    environment=env_config,
    agent_name="my_agent",
    device="cuda:0"
)
```

### Saving and Loading Configurations

Configurations can be easily saved to and loaded from JSON files:

```python
import json

# Save configuration to JSON
with open("agent_config.json", "w") as f:
    json.dump(agent_config.to_dict(), f, indent=2)

# Load configuration from JSON
with open("agent_config.json", "r") as f:
    config_dict = json.load(f)
    loaded_config = AgentConfig.from_dict(config_dict)
```

### Using Configuration Files with Command Line

You can specify a configuration file when running the main script:

```bash
python main.py --config_path configs/my_agent_config.json
```

### Example Configuration File

```json
{
  "agent_name": "red_teaming_agent_v1",
  "device": "cuda:0",
  "training": {
    "learning_rate": 5e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "max_epochs": 3
  },
  "model": {
    "base_model_path": "EleutherAI/pythia-1.4b",
    "use_lora": true,
    "lora_rank": 8,
    "lora_alpha": 16
  },
  "environment": {
    "dataset_path": "data/conversation_dataset.jsonl",
    "max_conversation_length": 512,
    "reward_threshold": 0.7
  }
}
```

For detailed examples of configuration usage, see `config_usage_examples.py`.

# Environment Details

#### env data flow: 
When the agent interacts with the environment, the environment passes a list of lists of strings, i.e. [["x1"],["x2"]]. The agent generates responses `"yi"` (via tokenizer.apply_chat_template() and helpers.build_chat()) and appends them to the state, creating [["x1","y1"],["x1","y1"]] and so on.

Each state is a list of strings representing the user and assistant chats in alternating fashion. Because both sources of text are LLMs, the "user" and "assistant" labels are applied adaptively so that the model that is performing model.generate(state) is the assistant.

#### env score computation
The score is computed via llama-guard model. The first token predicted by the guard contains probabilities for the words "safe" (ID 19193) and specific harm categories. The score is 1 - P("safe").

```
outputs = self.guard_model(input_ids, attention_mask=attention_mask).logits
logits = torch.softmax(outputs[:,0], dim=-1)
rewards = 1 - logits[:, 19193]
```

#### Token-wise Reward Feature
The token-wise reward feature measures the contribution of each token to the overall harm score of a sequence. This provides more fine-grained reward signals for training the policy.

Key components:
- **Token Masking**: Each token is individually masked to measure its marginal contribution to the harm score
- **Attention Divergence**: Changes in self-attention patterns when a token is masked are used to modulate the token's importance
- **Reward Attribution**: Token-level harm scores are normalized to sum to the sequence-level harm score

Implementation details:
- The harm critic uses token-level outputs (token_level=True) while the help critic uses sequence-level outputs (token_level=False)
- GAE advantages are always computed using sequence-level predicted rewards, even for the harm critic
- The first attention layer is cloned rather than directly modified to prevent updating the shared base model
- Token-wise rewards are scaled by the sequence-level GAE advantage to maintain proper magnitude

Benefits:
- More precise attribution of harm to specific tokens
- Better gradient signals for policy optimization
- Improved understanding of which parts of a response contribute to harmful content

Usage:
- The token-wise rewards are computed in the `value_step` function
- Only harm scores use token-wise rewards; help scores remain sequence-level
- Token-wise rewards are leveraged in policy gradients through token-level advantages

# Red Teaming Agent Evaluation

This repository contains code for evaluating the effectiveness of red teaming agents against language models.

## Overview

The evaluation framework allows for comparing different red teaming methods:
1. **Baseline**: Direct prompts from HarmBench dataset with no adversarial perturbations
2. **TemplateAgent**: Template-based red teaming with topic extraction and templated generation
3. **BTagent**: Bradley-Terry preference learning for template selection and optimization

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- HarmBench dataset

## Usage

The main evaluation script is `eval_agents.py`. It supports various modes of evaluation.

### Basic Usage

```bash
# Evaluate baseline direct prompts
python eval_agents.py \
  --agent_type baseline \
  --target_model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
  --guard_model_name_or_path "meta-llama/Llama-Guard-7b" \
  --behaviors_file "HarmBench/data/behavior_datasets/harmbench_behaviors_text_val.csv" \
  --num_behaviors 50 \
  --output_dir results/baseline

# Evaluate TemplateAgent
python eval_agents.py \
  --agent_type template \
  --target_model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
  --guard_model_name_or_path "meta-llama/Llama-Guard-7b" \
  --behaviors_file "HarmBench/data/behavior_datasets/harmbench_behaviors_text_val.csv" \
  --num_behaviors 50 \
  --agent_model_path "./models/template_agent" \
  --output_dir results/template_agent

# Evaluate BTagent
python eval_agents.py \
  --agent_type bt \
  --target_model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
  --guard_model_name_or_path "meta-llama/Llama-Guard-7b" \
  --behaviors_file "HarmBench/data/behavior_datasets/harmbench_behaviors_text_val.csv" \
  --num_behaviors 50 \
  --agent_model_path "./models/bt_agent" \
  --output_dir results/bt_agent
```

### Turn-Based Evaluation with Conversation History

For a more powerful evaluation that uses multiple turns and selects the best path:

```bash
python eval_agents.py \
  --agent_type bt \
  --target_model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
  --guard_model_name_or_path "meta-llama/Llama-Guard-7b" \
  --behaviors_file "HarmBench/data/behavior_datasets/harmbench_behaviors_text_val.csv" \
  --num_behaviors 50 \
  --agent_model_path "./models/bt_agent" \
  --output_dir results/bt_agent_turn_based \
  --turn_based \
  --concurrent_attempts 5 \
  --use_conversation_history
```

### Obfuscating History

If you want to obfuscate conversation history from the target model:

```bash
python eval_agents.py \
  --agent_type bt \
  --target_model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
  --guard_model_name_or_path "meta-llama/Llama-Guard-7b" \
  --behaviors_file "HarmBench/data/behavior_datasets/harmbench_behaviors_text_val.csv" \
  --num_behaviors 50 \
  --agent_model_path "./models/bt_agent" \
  --output_dir results/bt_agent_obfuscated \
  --turn_based \
  --concurrent_attempts 5 \
  --use_conversation_history \
  --obfuscate_history
```

## Output Format

The evaluation results are stored in the specified output directory:
- Individual JSON files for each behavior
- A summary JSON file with overall success rates
- A complete results.json containing all evaluation data

Example summary:
```json
{
  "total_behaviors": 50,
  "success_count": 32,
  "success_rate": 0.64,
  "total_attempts": 150,
  "successful_attempts": 38,
  "attempt_success_rate": 0.253,
  "attempts_per_behavior": 3,
  "use_conversation_history": true,
  "turn_based": true,
  "obfuscate_history": false
}
```

## Comparing Methods

To compare the effectiveness of different methods:

1. Run each method with the same behaviors dataset
2. Compare the success rates and attempt success rates
3. Analyze the conversation patterns in successful attempts to identify effective strategies

## Citation

If you use this code in your research, please cite:

```
@article{redteaming2023,
  title={Evaluating Red Teaming Agents on Large Language Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

# Token-wise Reward Function for Harm Prediction

## Overview

This implementation provides a token-wise reward function for more precise harm prediction. It calculates token-level contributions to harm scores using pairwise masking of the most important tokens (identified by attention values).

The approach:
1. Identifies top K tokens with highest attention values
2. Performs pairwise masking on these tokens
3. Calculates token contributions as the difference in predicted harm
4. Uses these contributions for more precise harm attribution

## Implementation

The implementation consists of the following files:

- `token_wise_reward.py`: Core implementation of the pairwise token masking approach
- `token_wise_reward_example.py`: Simple example of using the TokenWiseReward class

## Formulation

- $s - x_i$ = sequence $s$ with the ith token masked
- $x_i, x_j$ are tokens in the set of top K attended tokens in input $s$
