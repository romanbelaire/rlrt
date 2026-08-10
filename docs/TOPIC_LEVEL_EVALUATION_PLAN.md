# Topic-Level Red-Teaming Evaluation Plan

## Overview

This document outlines the plan to assess topic-level effectiveness of the adversarial agent in red-teaming experiments. The evaluation will partition the HarmBench dataset by semantic categories and measure attack success rates across different harm topics.

---

## Current Infrastructure Assessment

### 1. Red-Teaming Code Structure

**Training Components:**
- **Main Trainer**: `src/agent_trainer.py`, `src/mcq_agent_trainer.py`
- **Hierarchical Policies**:
  - High-level: `src/high_level_policy.py`, `src/hierarchical_policies.py` (template/instruction selection)
  - Low-level: `src/policies.py` (response generation)
- **Critics**: Sequence-level and token-level value critics
- **Environment**: `src/environment.py` (multi-step conversation environment)

**Evaluation Components:**
- **Test Runner**: `src/test_runner.py`
  - Loads trained agents from checkpoint directories
  - Evaluates on conversation-based attacks
  - Uses Llama-Guard-3-8B to detect harmful outputs
  - Logs conversations and computes success rates
  - Supports multiple evaluation scenarios (myopic/aware)

### 2. HarmBench Dataset

**Location**: `HarmBench/harmbench_behaviors_text_all.csv`

**Structure**:
- Total entries: 400
- Columns: `Behavior` (prompt), `SemanticCategory`, `FunctionalCategory`, `Tags`, `ContextString`, `BehaviorID`

**Topic Distribution** (SemanticCategory):
```
copyright                        100 (25.0%)
cybercrime_intrusion              67 (16.8%)
illegal                           65 (16.2%)
misinformation_disinformation     65 (16.2%)
chemical_biological               56 (14.0%)
harassment_bullying               25 (6.2%)
harmful                           22 (5.5%)
```

### 3. Saved Models

**Location**: `outputs/` directory

**Available Models**:
- `outputs/normal-agent-8b8t-std/run_20250801_195707/final_model/` ⭐ (Main model)
- `outputs/hl_only/run_20250821_105532/final_model/`
- `outputs/small_lora_chunk/` (multiple runs)
- `outputs/template-ablation-*/` (ablation studies)
- `outputs/normal-agent-8b8t-*` (variants: uniform, naive-reward, no-pairwise)

**Model Components** (per checkpoint):
- `config.json`: Full training configuration
- `policy/`: Low-level policy (LoRA adapters)
- `high_level_policy/`: High-level policy weights
- `sequence_critic/`: Sequence-level critic
- `token_critic/`: Token-level critic
- `metrics.json`: Training metrics

### 4. Existing Data Loaders

**HarmBench Loader**: `src/semantic_analysis/harmbench_loader.py`
- Already supports loading and preprocessing HarmBench CSV
- Has category filtering functions: `filter_harmbench_data()`, `sample_harmbench_data()`
- Handles SemanticCategory and FunctionalCategory columns

---

## Proposed Implementation Plan

### Phase 1: Create Topic-Level Test Runner

**File**: `src/topic_level_test_runner.py`

**Functionality**:
1. Load HarmBench dataset and partition by `SemanticCategory`
2. For each topic category:
   - Create topic-specific environment
   - Run full evaluation (aware_30steps scenario)
   - Collect per-topic metrics:
     - Success rate (harmful outputs / total attempts)
     - Average harm score (from guard model)
     - Average conversation length
     - Response quality metrics
3. Aggregate results across topics
4. Generate comprehensive report with:
   - Per-topic success rates
   - Comparison across topics
   - Statistical significance tests
   - Conversation examples per topic

**Key Features**:
- Reuse existing `test_runner.py` infrastructure
- Support batch evaluation across topics
- Save detailed logs per topic
- Generate JSON and CSV output formats

### Phase 2: Configuration Files

**File**: `configs/topic_level_eval_config.json`

**Structure**:
```json
{
  "model": {
    "model_name": "normal-agent-8b8t-std",
    "hf_checkpoint": "meta-llama/Llama-3.1-8B-Instruct",
    "use_high_level_policy": true,
    "high_level_policy_type": "ppo",
    "templates_path": "src/configs/templates.json"
  },
  "environment": {
    "data_path": "HarmBench/harmbench_behaviors_text_all.csv",
    "harm_threshold": 1.9,
    "max_steps": 30,
    "batch_size": 8,
    "max_new_tokens": 80,
    "target_model": "meta-llama/Llama-3.1-8B-Instruct",
    "guard_model": "meta-llama/Llama-Guard-3-8B"
  },
  "evaluation": {
    "episodes_per_topic": 20,
    "min_episodes_per_topic": 10,
    "eval_batch_size": 8,
    "scenario": "aware",
    "max_queries": 30,
    "save_conversations": true,
    "output_dir": "topic_level_results"
  },
  "topics": {
    "evaluate_all": true,
    "specific_topics": null,
    "stratify_by_frequency": true
  }
}
```

### Phase 3: Environment Modifications

**Modifications to** `src/environment.py`:

1. Add `category_filter` parameter to `ConversationDataset`
2. Support loading subsets by SemanticCategory
3. Ensure dataset balancing within topics

**New Method**:
```python
def filter_by_category(self, category: str) -> None:
    """Filter dataset to only include prompts from specified category."""
    if hasattr(self, 'df') and self.df is not None:
        self.df = self.df[self.df['SemanticCategory'] == category]
        self.questions = self.df['Behavior'].tolist()
```

### Phase 4: Results Analysis and Visualization

**File**: `scripts/analyze_topic_results.py`

**Features**:
1. Load results from all topic evaluations
2. Generate comparative visualizations:
   - Bar chart: Success rate by topic
   - Heatmap: Attack effectiveness across topics
   - Box plot: Harm score distributions
   - Line plot: Success rate vs. conversation length
3. Statistical analysis:
   - ANOVA test for significance across topics
   - Post-hoc pairwise comparisons
   - Confidence intervals
4. Export publication-ready tables and figures

### Phase 5: Execution Script

**File**: `scripts/run_topic_level_evaluation.sh`

**Usage**:
```bash
#!/bin/bash
# Run topic-level evaluation for all saved models

CHECKPOINT_DIR=$1
CONFIG_PATH="configs/topic_level_eval_config.json"
OUTPUT_DIR="topic_level_results/$(basename $CHECKPOINT_DIR)"

echo "Running topic-level evaluation for: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"

python src/topic_level_test_runner.py \
  --config-path $CONFIG_PATH \
  --checkpoint-dir $CHECKPOINT_DIR \
  --output-dir $OUTPUT_DIR \
  --eval-batch-size 8

echo "Analyzing results..."
python scripts/analyze_topic_results.py \
  --results-dir $OUTPUT_DIR \
  --output-format pdf

echo "Evaluation complete. Results saved to: $OUTPUT_DIR"
```

---

## Implementation Details

### Dataset Partitioning Strategy

**Option 1: Fixed Episodes Per Topic** (Recommended)
- Run same number of episodes for each topic (e.g., 20)
- Ensures fair comparison across topics
- May oversample small categories (harassment_bullying: 25 entries)
- May undersample large categories (copyright: 100 entries)

**Option 2: Proportional Sampling**
- Episodes proportional to category size
- More representative of overall dataset
- Harder to compare across topics (different N)

**Option 3: Stratified Sampling**
- Evaluate on all 400 prompts
- Group results by category post-hoc
- Most comprehensive but computationally expensive

**Recommendation**: Use Option 1 (fixed episodes per topic) with 20+ episodes per category for statistical power.

### Metrics to Track

**Primary Metrics**:
1. **Attack Success Rate (ASR)**: `harmful_outputs / total_episodes`
2. **Average Harm Score**: Mean guard model score for harmful outputs
3. **Jailbreak Efficiency**: `successful_attacks / total_queries`

**Secondary Metrics**:
4. **Average Conversation Length**: Mean number of turns to success
5. **First-Turn Success Rate**: Attacks successful on first message
6. **Template Effectiveness**: Which high-level templates work per topic
7. **Response Refusal Rate**: How often target refuses to respond

**Per-Topic Breakdowns**:
- Success rate distribution (histogram)
- Harm score vs. conversation length (scatter)
- Most/least effective attack strategies

### Output Format

**Directory Structure**:
```
topic_level_results/
├── normal-agent-8b8t-std/
│   ├── config.json                          # Evaluation config
│   ├── summary.json                         # Aggregated results
│   ├── results_by_topic.csv                 # Tabular results
│   ├── copyright/
│   │   ├── conversations.jsonl             # All conversations
│   │   ├── metrics.json                    # Topic metrics
│   │   └── examples.json                   # Top 5 successes
│   ├── cybercrime_intrusion/
│   │   └── ...
│   └── ...
├── visualizations/
│   ├── success_rate_by_topic.pdf
│   ├── harm_score_distribution.pdf
│   └── template_effectiveness.pdf
└── analysis_report.md                       # Human-readable report
```

**JSON Schema** (`summary.json`):
```json
{
  "model": "normal-agent-8b8t-std",
  "checkpoint": "outputs/normal-agent-8b8t-std/run_20250801_195707/final_model",
  "evaluation_date": "2025-10-08T12:00:00Z",
  "total_episodes": 140,
  "overall_asr": 0.67,
  "topics": {
    "copyright": {
      "episodes": 20,
      "success_count": 14,
      "asr": 0.70,
      "avg_harm_score": 0.78,
      "avg_conversation_length": 4.2,
      "first_turn_success_rate": 0.15
    },
    "cybercrime_intrusion": { ... },
    ...
  },
  "statistical_tests": {
    "anova_p_value": 0.023,
    "significant_differences": ["copyright vs harassment_bullying"]
  }
}
```

---

## Execution Timeline

### Week 1: Implementation
- **Day 1-2**: Create `topic_level_test_runner.py`
- **Day 3**: Modify `environment.py` for category filtering
- **Day 4**: Create configuration files
- **Day 5**: Test on small subset (1-2 topics, 5 episodes)

### Week 2: Evaluation
- **Day 6-7**: Run full evaluation on primary model (normal-agent-8b8t-std)
- **Day 8**: Run evaluation on ablation models (template-ablation-*)
- **Day 9**: Run evaluation on variant models (hl_only, no-pairwise)
- **Day 10**: Verify results and re-run if needed

### Week 3: Analysis
- **Day 11-12**: Implement `analyze_topic_results.py`
- **Day 13**: Generate visualizations and statistical tests
- **Day 14**: Write analysis report
- **Day 15**: Review and finalize

---

## Risk Assessment and Mitigation

### Computational Constraints
- **Risk**: 7 topics × 20 episodes × 30 queries = ~4200 forward passes per model
- **Mitigation**: 
  - Use batch processing (batch_size=8)
  - Enable GPU memory optimization
  - Run evaluations sequentially if memory constrained
  - Consider reducing max_queries to 20 for initial runs

### Data Imbalance
- **Risk**: Unequal topic sizes may bias results
- **Mitigation**:
  - Use fixed episodes per topic (Option 1)
  - Report confidence intervals
  - Use stratified sampling within topics

### Statistical Power
- **Risk**: Small categories (harmful: 22, harassment: 25) may have insufficient data
- **Mitigation**:
  - Sample with replacement for small categories
  - Report minimum detectable effect sizes
  - Consider grouping related categories

### Reproducibility
- **Risk**: Non-deterministic agent behavior may cause variability
- **Mitigation**:
  - Set random seeds in config
  - Run multiple seeds per topic (3-5 runs)
  - Report mean ± std dev

---

## Expected Outcomes

### Hypotheses to Test

**H1**: Attack success rate varies significantly across harm topics
- **Expected**: Yes (different topics have different complexity)
- **Measure**: ANOVA test, p < 0.05

**H2**: Technical topics (cybercrime, chemical_biological) are harder to attack
- **Expected**: Yes (require domain knowledge)
- **Measure**: Compare ASR for technical vs. non-technical topics

**H3**: Copyright and illegal topics have highest success rates
- **Expected**: Yes (more training data, less technical)
- **Measure**: Rank topics by ASR

**H4**: High-level policy adapts strategy based on topic
- **Expected**: Yes (different templates for different topics)
- **Measure**: Template usage distribution per topic

### Deliverables

1. **Code**:
   - `src/topic_level_test_runner.py`
   - `scripts/analyze_topic_results.py`
   - `scripts/run_topic_level_evaluation.sh`
   - `configs/topic_level_eval_config.json`

2. **Results**:
   - JSON summary file per model
   - CSV results table
   - Conversation logs per topic
   - Example successes/failures

3. **Analysis**:
   - Visualizations (PDF/PNG)
   - Statistical test results
   - Analysis report (Markdown)

4. **Documentation**:
   - This plan document
   - README for running evaluations
   - Interpretation guide for results

---

## Next Steps

1. **Review this plan** and confirm approach
2. **Select primary model** for initial evaluation (recommend: `normal-agent-8b8t-std`)
3. **Set computational budget** (GPUs available, time constraints)
4. **Begin implementation** with Phase 1 (topic-level test runner)
5. **Run pilot evaluation** on 1-2 topics to validate approach

---

## Questions for Discussion

1. Should we evaluate all models or focus on best-performing models?
2. What is the acceptable runtime per model? (est. 6-8 hours per model)
3. Should we include FunctionalCategory analysis in addition to SemanticCategory?
4. Do we need human evaluation of a subset to validate guard model judgments?
5. Should we track additional metrics (e.g., linguistic diversity, attack creativity)?

---

## References

- HarmBench Dataset: `HarmBench/harmbench_behaviors_text_all.csv`
- Test Runner: `src/test_runner.py`
- HarmBench Loader: `src/semantic_analysis/harmbench_loader.py`
- Configuration System: `src/config.py`
- Training Framework: `docs/HIERARCHICAL_RL_FRAMEWORK.md`

