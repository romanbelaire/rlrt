# MCQ Baseline Evaluator

This directory contains scripts to establish baseline performance of pre-trained language models on MCQ questions, for comparison with your trained MCQ agent.

## Files

- `baseline_mcq_evaluator.py` - Main baseline evaluator class
- `example_baseline_usage.py` - Example usage script
- `README_baseline.md` - This file

## Usage

### Command Line

```bash
# Basic usage with a model
python baseline_mcq_evaluator.py --model "microsoft/DialoGPT-medium"

# With custom dataset and output
python baseline_mcq_evaluator.py \
    --model "microsoft/DialoGPT-medium" \
    --dataset "path/to/dataset.json" \
    --output "results.json" \
    --temperature 0.0 \
    --max_tokens 5

# With HuggingFace dataset (ARC)
python baseline_mcq_evaluator.py \
    --model "microsoft/DialoGPT-medium" \
    --hf_dataset "allenai/ai2_arc" \
    --hf_subset "ARC-Challenge" \
    --hf_split "validation" \
    --hf_max_examples 100 \
    --output "arc_baseline_results.json"

# With HuggingFace dataset (HellaSwag)
python baseline_mcq_evaluator.py \
    --model "microsoft/DialoGPT-medium" \
    --hf_dataset "Rowan/hellaswag" \
    --hf_split "validation" \
    --hf_max_examples 100 \
    --output "hellaswag_baseline_results.json"

# Force CPU usage
python baseline_mcq_evaluator.py --model "microsoft/DialoGPT-medium" --device cpu
```

### Programmatic Usage

```python
from baseline_mcq_evaluator import MCQBaselineEvaluator

# Initialize evaluator
evaluator = MCQBaselineEvaluator(
    model_name="microsoft/DialoGPT-medium",
    device="auto",
    max_new_tokens=5,
    temperature=0.0
)

# Evaluate dataset
dataset = [
    {
        'question': 'What is the capital of France?',
        'choices': ['London', 'Berlin', 'Paris', 'Madrid'],
        'correct_answer': 2  # Paris (index 2)
    }
]

summary = evaluator.evaluate_dataset(dataset, "results.json")
print(f"Accuracy: {summary['accuracy']:.2%}")

# Cleanup
evaluator.cleanup()
```

## Supported Datasets

### HuggingFace Datasets

The evaluator directly supports these datasets with the same column structure as your training environment:

#### ARC (AI2 Reasoning Challenge)
- **Dataset ID**: `allenai/ai2_arc`
- **Subset**: `ARC-Challenge` (recommended)
- **Columns**: `question`, `choices.text`, `choices.label`, `answerKey`
- **Usage**: `--hf_dataset "allenai/ai2_arc" --hf_subset "ARC-Challenge"`

#### HellaSwag
- **Dataset ID**: `Rowan/hellaswag`
- **Columns**: `ctx`/`context`, `endings`, `label`
- **Usage**: `--hf_dataset "Rowan/hellaswag"`

### Custom Dataset Format

For custom datasets, use this JSON format:

```json
[
    {
        "question": "Question text here...",
        "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
        "correct_answer": 0
    }
]
```

Where `correct_answer` is the 0-based index of the correct choice.

### Dataset Loading Examples

```python
from baseline_mcq_evaluator import load_hf_dataset

# Load ARC dataset
arc_dataset = load_hf_dataset(
    dataset_id="allenai/ai2_arc",
    subset="ARC-Challenge",
    split="validation",
    max_examples=100
)

# Load HellaSwag dataset
hellaswag_dataset = load_hf_dataset(
    dataset_id="Rowan/hellaswag",
    split="validation",
    max_examples=100
)
```

## Templates

The evaluator uses the submission template from `src/configs/templates_general_hellaswag.json` by default. This template is designed to:

1. Present the question context clearly
2. Show all choices in A, B, C, D format
3. Request a single letter answer (A, B, C, or D)

## Output

Results are saved as JSON with:
- Summary statistics (accuracy, per-choice accuracy)
- Detailed results for each question
- Model configuration used

## Requirements

- `torch`
- `transformers`
- `tqdm`
- `numpy`

## Example Output

```
EVALUATION SUMMARY
============================================================
Model: microsoft/DialoGPT-medium
Total Questions: 100
Correct Answers: 75
Accuracy: 0.7500 (75.00%)

Per-Choice Accuracy:
  A: 0.8000 (80.00%)
  B: 0.7000 (70.00%)
  C: 0.7500 (75.00%)
  D: 0.7500 (75.00%)
============================================================
```
