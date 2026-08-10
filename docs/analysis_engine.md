# Semantic Analysis Engine

## Overview

The Semantic Analysis Engine is a **completely optional** analysis tool that provides deep insights into model behavior during training. It's designed as a separate analysis pipeline, not integrated into the core training loop, allowing you to easily enable or disable it based on your needs.

## What It Does

The analysis engine computes and tracks:

- **Semantic Drift Metrics**: How model responses change over time
- **Guard Model Scores**: Safety evaluations using LlamaGuard
- **Vector Similarities**: Embedding-based analysis of conversation patterns
- **Behavioral Consistency**: Track how model responses evolve across training episodes

## Key Components

### 1. Analysis Engine (`src/semantic_analysis/analysis_engine.py`)
- Main orchestrator for the semantic drift analysis pipeline
- Handles conversation loading, metric computation, and output generation
- **Important**: Modifies metrics in-place for JSON serialization (this was the source of the list conversion bug)

### 2. Drift Calculator (`src/semantic_analysis/drift_calculator.py`)
- Computes 20 different semantic drift metrics
- Supports both hidden-state and input embedding analysis
- Generates similarity scores between different conversation turns

### 3. Vector Store (`src/semantic_analysis/vector_store.py`)
- FAISS-based storage for conversation embeddings
- Enables similarity search across training episodes
- Memory-intensive for long training runs

### 4. Data Collector (`src/semantic_analysis/collector.py`)
- Gathers conversation data during training
- Stores embeddings and metadata for analysis

## Configuration

The semantic analysis is configured through `SemanticAnalysisConfig` in `src/config.py`:

```python
@dataclass
class SemanticAnalysisConfig:
    # Collection settings
    max_conversation_length: int = 30
    batch_size: int = 4
    save_interval: int = 100
    
    # Vector storage settings
    faiss_index_type: str = "IndexFlatIP"
    embedding_dim: int = 4096
    
    # Output settings
    output_dir: str = "semantic_analysis_output"
    vector_store_path: str = "semantic_vectors"
    
    # Model settings
    target_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    guard_model: str = "meta-llama/Llama-Guard-3-8B"
    use_quantization: bool = True
```

## Usage Modes

### 1. **Disabled (Recommended for Prototyping)**
```python
# In your config file, either omit semantic_analysis entirely or set:
semantic_analysis: null
```

**Benefits:**
- Faster training (no tensor-to-list conversions)
- Lower memory usage (no vector stores)
- Cleaner code execution
- Focus on core training functionality

### 2. **Enabled for Analysis**
```python
# In your config file:
semantic_analysis:
  max_conversation_length: 30
  batch_size: 4
  output_dir: "semantic_analysis_output"
```

**Benefits:**
- Deep insights into model behavior
- Semantic drift tracking
- Safety evaluation metrics
- Vector similarity analysis

## How to Run

### Training (Semantic Analysis Disabled)
```bash
python src/main.py --config your_config.yaml
```
This runs training without any semantic analysis overhead.

### Semantic Analysis Only (After Training)
```bash
python src/semantic_analysis/evaluate_semantic_drift.py \
  --data_dir your_training_outputs \
  --output_dir semantic_analysis_results \
  --checkpoint_dir your_model_checkpoints
```

### Data Collection During Training
```bash
python src/semantic_analysis/main.py \
  --config your_config.yaml \
  --checkpoint_dir your_model_checkpoints
```

## Memory and Performance Impact

### When Disabled
- **Memory**: Minimal overhead
- **Speed**: No impact on training
- **Storage**: No additional files generated

### When Enabled
- **Memory**: Significant (embeddings + vector stores)
- **Speed**: Slower due to tensor processing
- **Storage**: Large files (embeddings, metrics, visualizations)

## Known Issues and Fixes

### List Conversion Bug (Fixed)
**Problem**: The analysis engine was converting tensor values to lists for JSON serialization, causing CSV writing to fail with `TypeError: float() argument must be a string or a number, not 'list'`.

**Root Cause**: In-place modification of the metrics dictionary in `save_metrics_to_jsonl()`.

**Fix**: Create a copy for JSON serialization instead of modifying the original:
```python
# Before (buggy):
metrics[key] = [v.detach().cpu().float().numpy().tolist() for v in value]

# After (fixed):
json_metrics[key] = [v.detach().cpu().float().numpy().tolist() for v in value]
```

## Best Practices

### For Prototyping
1. **Disable semantic analysis** by omitting it from config
2. Focus on core training metrics and performance
3. Use standard logging and CSV outputs

### For Research/Production
1. **Enable semantic analysis** after prototype is working
2. Monitor memory usage and storage requirements
3. Consider downsampling for long training runs
4. Use separate storage for vector databases

### For Debugging
1. Start with semantic analysis disabled
2. Enable only when you need specific insights
3. Monitor the `semantic_analysis_output` directory size

## Output Files

When enabled, the analysis engine generates:

```
semantic_analysis_output/
├── data/
│   └── semantic_drift_metrics.jsonl
├── metadata/
│   └── semantic_drift_metrics.csv
├── plots/
│   ├── similarity_heatmaps.png
│   ├── drift_trajectories.png
│   └── guard_scores.png
└── reports/
    └── summary_report.md
```

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce `batch_size` or disable vector storage
2. **Slow performance**: Use quantization or reduce embedding dimensions
3. **Storage full**: Monitor output directory size and clean old runs

### Performance Tuning
- Set `use_quantization: true` for faster inference
- Reduce `embedding_dim` if memory is constrained
- Increase `save_interval` to reduce I/O overhead

## Summary

The Semantic Analysis Engine is a powerful but optional tool for understanding model behavior during training. For prototyping and development:

- **Keep it disabled** to focus on core functionality
- **Enable only when needed** for research insights
- **Monitor resource usage** when enabled
- **Use separate analysis runs** to avoid training overhead

This design allows you to iterate quickly on your training setup while maintaining the option for deep behavioral analysis when your prototype is working. 