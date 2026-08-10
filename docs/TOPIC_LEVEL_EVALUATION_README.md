# Topic-Level Red-Teaming Evaluation

## Quick Start

This guide will help you run topic-level effectiveness evaluation on your trained red-teaming agents.

## Overview

The topic-level evaluation system partitions the HarmBench dataset by semantic categories (topics) and evaluates your adversarial agent's effectiveness on each topic separately. This allows you to:

- Identify which harm topics your agent is most/least effective at attacking
- Compare performance across different semantic categories
- Understand topic-specific attack strategies
- Generate comprehensive reports with visualizations and statistical analysis

## Files Created

### Core Implementation
- `src/topic_level_test_runner.py` - Main evaluation script
- `scripts/analyze_topic_results.py` - Analysis and visualization script
- `scripts/run_topic_level_evaluation.sh` - Convenient wrapper script
- `configs/topic_level_eval_config.json` - Configuration template

### Documentation
- `docs/TOPIC_LEVEL_EVALUATION_PLAN.md` - Detailed implementation plan
- `docs/TOPIC_LEVEL_EVALUATION_README.md` - This file

## HarmBench Topics

The dataset includes 7 semantic categories (400 total prompts):

1. **copyright** (100 prompts, 25%) - Copyright infringement content
2. **cybercrime_intrusion** (67 prompts, 17%) - Hacking, intrusion, cyber attacks
3. **illegal** (65 prompts, 16%) - General illegal activities
4. **misinformation_disinformation** (65 prompts, 16%) - False information
5. **chemical_biological** (56 prompts, 14%) - Dangerous substances/weapons
6. **harassment_bullying** (25 prompts, 6%) - Harassment and bullying
7. **harmful** (22 prompts, 6%) - Other harmful content

## Usage

### Method 1: Using the Wrapper Script (Recommended)

```bash
# Evaluate all topics on your trained model
./scripts/run_topic_level_evaluation.sh \
    outputs/normal-agent-8b8t-std/run_20250801_195707/final_model

# Evaluate specific topics only
./scripts/run_topic_level_evaluation.sh \
    outputs/normal-agent-8b8t-std/run_20250801_195707/final_model \
    --topics copyright cybercrime_intrusion illegal

# Custom configuration
./scripts/run_topic_level_evaluation.sh \
    outputs/normal-agent-8b8t-std/run_20250801_195707/final_model \
    --episodes 50 \
    --max-queries 40 \
    --batch-size 4
```

### Method 2: Direct Python Invocation

```bash
# Run evaluation
python src/topic_level_test_runner.py \
    --config-path configs/topic_level_eval_config.json \
    --checkpoint-dir outputs/normal-agent-8b8t-std/run_20250801_195707/final_model \
    --output-dir topic_level_results/normal-agent-8b8t-std \
    --eval-batch-size 8 \
    --episodes-per-topic 20 \
    --max-queries 30

# Run analysis separately
python scripts/analyze_topic_results.py \
    --results-dir topic_level_results/normal-agent-8b8t-std \
    --output-format pdf \
    --include-statistical-tests
```

## Command Line Options

### Evaluation Options (`topic_level_test_runner.py`)

- `--config-path PATH` - Base configuration file (required)
- `--checkpoint-dir PATH` - Model checkpoint directory (required)
- `--output-dir PATH` - Output directory for results (required)
- `--eval-batch-size N` - Batch size for evaluation (default: 8)
- `--episodes-per-topic N` - Episodes per topic (default: 20)
- `--max-queries N` - Max queries per episode (default: 30)
- `--topics TOPIC1 TOPIC2 ...` - Specific topics to evaluate (default: all)

### Analysis Options (`analyze_topic_results.py`)

- `--results-dir PATH` - Results directory (required)
- `--output-format {pdf,png}` - Figure format (default: pdf)
- `--include-statistical-tests` - Include ANOVA and pairwise tests
- `--no-statistical-tests` - Skip statistical tests

## Output Structure

After running evaluation, your output directory will contain:

```
topic_level_results/
└── normal-agent-8b8t-std/
    ├── config.json                          # Evaluation configuration
    ├── summary.json                         # Aggregated results
    ├── results_by_topic.csv                 # Tabular results
    ├── analysis_report.md                   # Human-readable report
    ├── statistical_tests.json               # ANOVA and pairwise results
    ├── pairwise_comparisons.csv             # All pairwise comparisons
    ├── visualizations/
    │   ├── success_rate_by_topic.pdf        # Bar chart of ASR by topic
    │   ├── harm_score_distribution.pdf      # Harm scores by topic
    │   ├── conversation_length_vs_asr.pdf   # Efficiency scatter plot
    │   ├── pairwise_heatmap.pdf             # Significance heatmap
    │   └── metric_correlations.pdf          # Correlation matrix
    ├── copyright/
    │   ├── conversations.jsonl              # All conversations
    │   ├── metrics.json                     # Topic-specific metrics
    │   └── ...
    ├── cybercrime_intrusion/
    │   └── ...
    └── ...
```

## Key Metrics

### Primary Metrics

1. **Attack Success Rate (ASR)**: Fraction of episodes that successfully elicit harmful output
2. **Average Harm Score**: Mean guard model score for harmful outputs (0-1 scale)
3. **Jailbreak Efficiency**: Successful attacks per query

### Secondary Metrics

4. **Average Conversation Length**: Mean turns required for success
5. **First-Turn Success Rate**: Attacks successful on first message
6. **Template Effectiveness**: Which high-level strategies work per topic
7. **Response Refusal Rate**: How often target refuses

## Example Workflow

### 1. Initial Evaluation

Run full evaluation on your primary model:

```bash
./scripts/run_topic_level_evaluation.sh \
    outputs/normal-agent-8b8t-std/run_20250801_195707/final_model \
    --episodes 20
```

**Expected Time**: ~6-8 hours (depends on GPU and model)

### 2. Review Results

Check the generated report:

```bash
cat topic_level_results/normal-agent-8b8t-std/analysis_report.md
```

View visualizations:

```bash
ls topic_level_results/normal-agent-8b8t-std/visualizations/
```

### 3. Targeted Follow-up

If specific topics show unexpected results, run focused evaluation:

```bash
./scripts/run_topic_level_evaluation.sh \
    outputs/normal-agent-8b8t-std/run_20250801_195707/final_model \
    --topics chemical_biological cybercrime_intrusion \
    --episodes 50 \
    --max-queries 40
```

### 4. Compare Models

Run evaluation on multiple models:

```bash
for model_dir in outputs/*/run_*/final_model; do
    model_name=$(basename $(dirname $(dirname $model_dir)))
    echo "Evaluating: $model_name"
    ./scripts/run_topic_level_evaluation.sh $model_dir \
        --output-dir topic_level_results/$model_name
done
```

Then compare results across models using the CSV files.

## Configuration

### Modifying Evaluation Parameters

Edit `configs/topic_level_eval_config.json`:

```json
{
  "environment": {
    "max_steps": 30,           // Increase for longer conversations
    "harm_threshold": 1.9      // Adjust harm detection threshold
  },
  "evaluation": {
    "episodes_per_topic": 20,  // More episodes = better statistics
    "max_queries": 30          // Budget per episode
  }
}
```

### Topic Selection

To evaluate only specific topics, use the `--topics` flag:

```bash
# Technical topics only
./scripts/run_topic_level_evaluation.sh MODEL_DIR \
    --topics chemical_biological cybercrime_intrusion

# Non-technical topics only
./scripts/run_topic_level_evaluation.sh MODEL_DIR \
    --topics copyright harassment_bullying misinformation_disinformation
```

## Troubleshooting

### Out of Memory Errors

**Problem**: GPU runs out of memory during evaluation

**Solution**: Reduce batch size
```bash
./scripts/run_topic_level_evaluation.sh MODEL_DIR --batch-size 4
```

### Slow Evaluation

**Problem**: Evaluation takes too long

**Solutions**:
1. Reduce episodes per topic: `--episodes 10`
2. Reduce max queries: `--max-queries 20`
3. Use GPU with more memory
4. Evaluate specific topics only: `--topics copyright illegal`

### Missing Dependencies

**Problem**: Import errors when running scripts

**Solution**: Install required packages
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Checkpoint Not Found

**Problem**: Cannot find model checkpoint

**Solution**: Verify checkpoint structure
```bash
ls outputs/normal-agent-8b8t-std/run_*/final_model/
# Should contain: policy/, high_level_policy/, config.json
```

## Statistical Interpretation

### ANOVA Results

- **p < 0.05**: Significant differences exist between topics
- **p ≥ 0.05**: No significant differences (consistent performance)

### Pairwise Comparisons

The analysis includes Bonferroni correction for multiple comparisons:
- Use `p_value_bonferroni` for conservative significance testing
- Use `p_value` (uncorrected) for exploratory analysis

### Confidence Intervals

95% confidence intervals use the Clopper-Pearson method for binomial proportions:
- Wider intervals indicate less certainty (fewer samples or high variance)
- Non-overlapping intervals suggest significant differences

## Advanced Usage

### Custom Analysis

Load results programmatically:

```python
import json
import pandas as pd

# Load summary
with open('topic_level_results/model/summary.json') as f:
    summary = json.load(f)

# Load per-topic results
df = pd.read_csv('topic_level_results/model/results_by_topic.csv')

# Custom analysis
print(f"Overall ASR: {summary['overall_asr']:.3f}")
print(f"Best topic: {df.loc[df['asr'].idxmax(), 'topic']}")
print(f"Worst topic: {df.loc[df['asr'].idxmin(), 'topic']}")
```

### Batch Evaluation Script

Create a script to evaluate multiple models:

```bash
#!/bin/bash
# batch_evaluate.sh

MODELS=(
    "outputs/normal-agent-8b8t-std/run_20250801_195707/final_model"
    "outputs/hl_only/run_20250821_105532/final_model"
    "outputs/template-ablation-0/run_20250717_035924/final_model"
)

for MODEL in "${MODELS[@]}"; do
    echo "Evaluating: $MODEL"
    ./scripts/run_topic_level_evaluation.sh "$MODEL" || echo "Failed: $MODEL"
done
```

## Performance Expectations

### Computational Requirements

- **GPU Memory**: ~20GB (target + guard models)
- **Time per Topic**: ~30-60 minutes (20 episodes)
- **Total Time (7 topics)**: ~6-8 hours
- **Disk Space**: ~1-2GB per model (logs + results)

### Recommended Resources

- **Minimum**: 1x NVIDIA A40 (48GB)
- **Optimal**: 2x GPUs (separate agent and target/guard)
- **CPU**: 16+ cores recommended for batch processing

## Next Steps

1. **Run Initial Evaluation**: Start with default settings on your best model
2. **Review Results**: Check `analysis_report.md` for insights
3. **Identify Patterns**: Look for topics with unexpected performance
4. **Iterate**: Run focused evaluations on interesting topics
5. **Compare Models**: Evaluate multiple models to understand improvements

## Support

For issues or questions:

1. Check `docs/TOPIC_LEVEL_EVALUATION_PLAN.md` for detailed design
2. Review existing `src/test_runner.py` for base evaluation logic
3. Examine conversation logs in `<output_dir>/<topic>/` for debugging

## References

- **HarmBench Dataset**: `HarmBench/harmbench_behaviors_text_all.csv`
- **Base Test Runner**: `src/test_runner.py`
- **Configuration System**: `src/config.py`
- **Environment**: `src/environment.py`

---

**Last Updated**: 2025-10-08

