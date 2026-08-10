# Semantic Analysis Data Collection

This module implements semantic analysis data collection for HarmBench conversations, recording detailed metrics including LlamaGuard scores, vector embeddings, and hierarchical policy scores.

## Overview

The semantic analysis system collects comprehensive data from red team conversations with the following features:

- **N=30 turn conversations** per HarmBench data point
- **LlamaGuard safety scores** for target responses
- **Vector embeddings** for actions, responses, and conversation context
- **Hierarchical policy scores** (V_high and Q_low)
- **FAISS-based vector storage** for efficient similarity search
- **Category-based analysis** from HarmBench data

## Data Collection Requirements

For each conversation, the system records:

1. **Original question** (one item per conversation)
2. **Text sequences** (seq length N=30):
   - High-level instructions
   - Low-level actions  
   - Target responses
3. **LlamaGuard scores** for target responses (seq length N=30)
4. **Vector data** (seq length N=30):
   - Action embeddings
   - Response embeddings
   - Cumulative conversation context embeddings
5. **Category** of original question (one item per conversation)
6. **V_high scores** for each action (seq length N=30)
7. **Mean Q_low scores** for each action (seq length N=30)

## Installation

The module requires the following dependencies:

```bash
pip install faiss-cpu torch pandas tqdm scikit-learn seaborn matplotlib
```

For GPU support:
```bash
pip install faiss-gpu
```

## Usage

### Collect Data

```bash
python src/semantic_analysis/main.py --checkpoint_dir /path/to/checkpoint --config_path configs/semantic_analysis_config.json
```

### Evaluate Semantic Drift (this script performs the analysis in this README’s evaluation spec)

```bash
python src/semantic_analysis/evaluate_semantic_drift.py \
  --data_dir semantic_analysis_output \
  --downsample_dim 256 \
  --max_conversations 100
```

- The script will:
  - Load conversations from `--data_dir` or `--data_dir/conversations`.
  - Down-sample all stored embeddings (actions, responses, contexts, and original prompt) to `--downsample_dim` using PCA (required).
  - Compute the 20 semantic drift metrics: 4 context voices × 5 context ranges.
  - Save a JSONL plus a CSV of all metrics under `--output_dir` (defaults to `--data_dir`).
  - Generate plots of each metric vs time and vs running-max of harm metrics (R_{0:t}) for LlamaGuard, V_high, Q_low.
  - Write a markdown summary with per-metric mean/std.

### Command Line Interface (collector)

```bash
# Basic data collection
python src/semantic_analysis/main.py --checkpoint_dir /path/to/checkpoint

# With custom configuration
python src/semantic_analysis/main.py --checkpoint_dir /path/to/checkpoint --config_path configs/my_config.json

# Limit number of conversations
python src/semantic_analysis/main.py --checkpoint_dir /path/to/checkpoint --max_conversations 100

# Sample data for testing
python src/semantic_analysis/main.py --checkpoint_dir /path/to/checkpoint --sample_size 50

# Filter by specific categories
python src/semantic_analysis/main.py --checkpoint_dir /path/to/checkpoint --categories cybersecurity privacy

# Dry run to check data loading
python src/semantic_analysis/main.py --checkpoint_dir /path/to/checkpoint --dry_run

# Custom output directory
python src/semantic_analysis/main.py --checkpoint_dir /path/to/checkpoint --output_dir my_output
```

### Evaluation Details: 20 Metrics

We define semantic drift as cosine similarity between two states X and Y. The four context voices define how X and Y are constructed per turn t:

- 1a (agent only): X = mean of agent actions up to t-1; Y = agent action at t.
- 1b (target only): X = mean of target responses up to t-1; Y = target response at t.
- 1c (both, agent in X): X = mean of actions up to and including t mixed with mean of responses up to t-1; Y = the pair/context at t.
- 1d (both, agent in Y): X = mean of previous pairs up to t-1; Y = ONLY the response at t.

Context ranges to compare for each voice:

- p2c: similarity(X_t, Y_t)
- p2p: similarity(Y_{t-1}, Y_t) (None at t=0)
- p2c_cu: cumulative sum over time of p2p
- p2o: similarity(original_prompt, Y_t)
- p2f: similarity(Y_0, Y_t) with Y_0 defined per voice

### Output Structure

The analysis produces the following in `--output_dir` (defaults to `--data_dir`):

```
<output_dir>/
├── data/
│   └── semantic_drift_metrics.jsonl
├── metadata/
│   └── semantic_drift_metrics.csv
├── plots/
│   ├── similarity_1a_p2c_vs_time.png
│   ├── similarity_1a_p2c_vs_max_llamaguard_to_t.png
│   └── ...
└── reports/
    └── analysis_summary.md
```

### Programmatic Usage

```python
from src.config import AgentConfig
from src.semantic_analysis.collector import SemanticDataCollector
from src.semantic_analysis.harmbench_loader import load_harmbench_data

# Load configuration
config = AgentConfig.from_json("configs/semantic_analysis_config.json")

# Load HarmBench data
harmbench_data = load_harmbench_data(config.semantic_analysis)

# Initialize collector
collector = SemanticDataCollector(config, checkpoint_dir="/path/to/checkpoint")

# Collect data
collector.collect_dataset(harmbench_data, max_conversations=100)
```

## Testing

Run the test suite to verify the implementation:

```bash
python src/semantic_analysis/test_implementation.py
```

This will test:
- Data structure serialization/deserialization
- Vector store functionality
- Conversation processing
- Configuration handling

## Performance Considerations

### Memory Management

- Process conversations in batches (default: 4)
- Clear GPU cache between batches
- Use gradient checkpointing for large models

### Storage Optimization

- Compress embeddings before FAISS storage
- Use efficient serialization formats
- Implement incremental saving

### Parallel Processing

- Multi-GPU support for embedding extraction
- Parallel conversation processing
- Asynchronous I/O for storage operations

## Error Handling

The system follows fail-fast and loud principles in this research codebase.

## Future Extensions

- Additional metrics
- Advanced analysis (clustering, temporal policy evolution)
- Real-time processing

## Troubleshooting

- Ensure FAISS installation matches your environment (CPU/GPU)
- Reduce batch size if OOM
- Check HarmBench CSV path and schema

## API Reference

- `SemanticDataCollector.collect_dataset`
- `SemanticVectorStore.add_conversation_vectors`
- `evaluate_semantic_drift.py` entrypoint for analysis 