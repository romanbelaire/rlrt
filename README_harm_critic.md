# Harm Critic Training

This module provides functionality to train a standalone harm value critic model using HarmBench (harmful examples) and OASST1 (safe examples) datasets.

## Overview

The harm critic is designed to predict harmfulness scores for text inputs at both the sequence and token levels. It can be trained independently from the full agent, making it easier to develop and evaluate harm detection capabilities.

Key features:
- **Token-level Harm Attribution**: Uses masked token analysis to identify which tokens contribute most to harmful content
- **Attention Divergence Mechanism**: Encourages the model to develop more differentiated attention patterns for better interpretability
- **Token-wise Loss Weighting**: Weights token contributions by their attention divergence for more focused learning
- **Binary Classification Training**: Uses HarmBench examples (label 1.0) and OASST1 examples (label 0.0) for training
- **LoRA Adapter Support**: Efficiently fine-tune specific attention layers with minimal memory requirements
- **Modular Architecture**: Works with any Transformer model from Hugging Face

## Training Process

The training process uses a token masking approach to identify the contribution of individual tokens to the overall harmfulness score. When combined with attention divergence loss, this enables the model to learn which tokens contribute most to harmful content.

The workflow is:

1. Load datasets from HarmBench (harmful) and OASST1 (safe)
2. For each training example:
   - Make a baseline prediction
   - Mask individual tokens and observe difference in predictions
   - Calculate attention divergence between original and masked patterns
   - These differences determine token-level contributions
3. Calculate losses:
   - Sequence-level MSE loss for overall prediction accuracy
   - Token-wise MSE loss that weights token contributions by their attention divergence
   - KL divergence loss for attention patterns to enhance interpretability
4. Update model parameters to minimize combined losses
5. Evaluate on validation set and save checkpoints

### Token-wise Loss Calculation

A key innovation in this training process is the token-wise loss calculation that leverages attention divergence:

1. For each token position N in a sequence, we compute:
   - The token's contribution to the overall harm score (difference in prediction when token N is masked)
   - The attention divergence between the original sequence and the sequence with token N masked
2. The token-wise loss computes MSE between:
   - The weighted token contributions (contribution * divergence) 
   - Target token values (label distributed evenly across sequence length)

This approach amplifies the gradient magnitude for tokens that both:
- Significantly affect the prediction (high contribution)
- Create distinct attention patterns (high divergence)

This encourages the model to focus on truly harmful tokens and properly distinguish their impact from neutral content.

## LoRA Adapter Training

The harm critic now supports Low-Rank Adaptation (LoRA) for efficient fine-tuning. This approach:

- Reduces memory requirements by up to 90% compared to full fine-tuning
- Allows targeting specific attention layers (default: layer 0)
- Enables compute_attention_divergence to run on GPU even with large models
- Provides faster training iterations and better scalability
- Results in compact adapter weights (~2-10MB) instead of full model checkpoints

LoRA works by adding trainable low-rank matrices to key attention components while keeping the base model frozen. This dramatically reduces the number of trainable parameters while achieving similar performance to full fine-tuning.

## Usage

### Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- PEFT library (Parameter-Efficient Fine-Tuning)
- Datasets library
- HarmBench dataset (place in `./HarmBench` directory)
- Internet connection to download OASST1 dataset

### Training a Model

To train a model using default parameters with LoRA adapters:

```bash
python train_value.py
```

To customize training parameters with LoRA:

```bash
python train_value.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --harmful_samples 2000 \
  --safe_samples 2000 \
  --num_epochs 10 \
  --attention_divergence_weight 0.2 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_layers 0 1 2
```

For full-parameter fine-tuning (not recommended for large models):

```bash
python train_value.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --batch_size 4 \
  --learning_rate 5e-6 \
  --num_epochs 3 \
  --use_lora false
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Base model to use for ValueTransformer | meta-llama/Llama-3.2-1B-Instruct |
| `--checkpoints_dir` | Directory to save checkpoints | checkpoints/harm_critic |
| `--batch_size` | Training batch size | 8 |
| `--accumulation_steps` | Gradient accumulation steps | 4 |
| `--learning_rate` | Learning rate for optimizer | 2e-5 |
| `--attention_divergence_weight` | Weight for attention divergence loss | 0.1 |
| `--num_epochs` | Number of epochs to train | 5 |
| `--harmful_samples` | Number of harmful samples from HarmBench | 1000 |
| `--safe_samples` | Number of safe samples from OASST1 | 1000 |
| `--use_lora` | Whether to use LoRA adapters | true |
| `--lora_r` | LoRA rank parameter | 8 |
| `--lora_alpha` | LoRA alpha parameter | 16 |
| `--lora_dropout` | LoRA dropout rate | 0.05 |
| `--target_modules` | Module types to apply LoRA to | q_proj v_proj |
| `--target_layers` | Attention layer indices to apply LoRA to | 0 |

### Loading a Trained Model

To use a trained model with LoRA adapters:

```python
import torch
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from models import ValueTransformer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Method 1: Using saved checkpoint directly
checkpoint = torch.load("checkpoints/harm_critic/harm_critic_best.pt")

# Initialize base model
base_model = ValueTransformer(
    hf_checkpoint="meta-llama/Llama-3.2-1B-Instruct",
    tokenizer=tokenizer,
    token_level=True,
    attention_update=True
)

# Load LoRA configuration
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=checkpoint['lora_config']['r'],
    lora_alpha=checkpoint['lora_config']['alpha'],
    target_modules=checkpoint['lora_config']['target_modules'],
    lora_dropout=checkpoint['lora_config']['dropout'],
    bias=checkpoint['lora_config']['bias'],
    task_type="SEQ_CLS"
)

# Apply LoRA to base model
model = get_peft_model(base_model, lora_config)

# Load the adapter weights
model.load_state_dict(checkpoint['adapter_model'])

# Use the model
text = "Is this text harmful?"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    harm_score, token_scores = model(
        input_ids=inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        return_token_scores=True
    )
    
print(f"Overall harm score: {harm_score.item():.4f}")
```

## Files

- `value_trainer.py`: Main class implementing the training logic
- `train_value.py`: Command-line script for training
- `README_harm_critic.md`: This documentation

## Integration with Agent

The trained harm critic can be integrated with the full agent by loading the checkpoint:

```python
from agent import Agent
from models import ValueTransformer
from peft import LoraConfig, get_peft_model
import torch

# Initialize agent
agent = Agent(...)

# For LoRA-trained models
checkpoint = torch.load("checkpoints/harm_critic/harm_critic_best.pt")

# Initialize base harm critic
base_harm_critic = ValueTransformer(
    hf_checkpoint=checkpoint['base_model_name'],
    tokenizer=agent.tokenizer,
    token_level=checkpoint['config']['token_level'],
    attention_update=True
)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=checkpoint['lora_config']['r'],
    lora_alpha=checkpoint['lora_config']['alpha'],
    target_modules=checkpoint['lora_config']['target_modules'],
    lora_dropout=checkpoint['lora_config']['dropout'],
    bias=checkpoint['lora_config']['bias'],
    task_type="SEQ_CLS"
)

# Create model with LoRA adapters
agent.harm_critic = get_peft_model(base_harm_critic, lora_config)

# Load adapter weights
agent.harm_critic.load_state_dict(checkpoint['adapter_model'])
```

## Future Work

- Support for multi-class harm classification (beyond binary harmful/safe)
- Integration with other datasets like ToxicChat, HarmfulQA
- Ensemble methods combining multiple harm critics
- Calibration techniques for better threshold settings
- Extending LoRA adapters to other parts of the model 