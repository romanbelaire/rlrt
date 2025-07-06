# GPU Memory Management Optimizations

This document describes the memory optimizations implemented to prevent CUDA out of memory errors during training.

## Problem Analysis

The original error occurred when transitioning from warmup (short sequences) to full training (long sequences):
- GPU 0 was using 91.81 GiB out of 93.10 GiB available memory (98% utilization)
- Trying to allocate additional 1.39 GiB for actor training caused OOM
- The issue was at the `actor.get_log_probs()` call in `train_actor()`

## Implemented Solutions

### 1. Configuration Optimizations

**Memory Settings in `src/configs/config.json`:**
```json
{
  "training": {
    "max_gpu_memory": 0.25,          // Reduced from 0.4 to 0.25 (25% vs 40%)
    "dynamic_batch_sizing": true,    // Enable dynamic batch sizing
    "max_memory_per_gpu": 0.75,      // Target 75% GPU memory usage
    "chunked_forward_pass": true,    // Enable chunked processing
    "chunk_size": 2,                 // Reduced chunk size from 4 to 2
    "use_first_layer_attention_only": true,
    
    // Critic chunking optimizations
    "critic_chunking_threshold": 50, // Reduced from 100
    "critic_max_chunk_size": 1,      // Reduced from 2 to 1
    "critic_memory_fraction": 0.4,   // Reduced from 0.6 to 0.4
    
    // New actor chunking settings
    "actor_chunking_enabled": true,
    "actor_max_chunk_size": 2,
    "actor_memory_fraction": 0.5
  },
  "model": {
    "target_layers": [0, 1, 2]       // Reduced LoRA layers from [0,1,2,3,4] to [0,1,2]
  }
}
```

### 2. Actor Chunking Implementation

**New chunked actor training (`src/agent_trainer.py`):**
- Added `_train_actor_chunked()` method that processes batches in smaller chunks
- Automatic chunking trigger when `batch_size * seq_len > 2048` or chunking is enabled
- Memory-aware chunk size calculation using `get_optimal_chunk_size()`
- Proper gradient accumulation across chunks
- Aggressive memory cleanup with `torch.cuda.empty_cache()`

### 3. PyTorch Memory Configuration

**Environment Variables:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

Use the provided `run_with_memory_optimizations.py` script to automatically set these.

## Memory Optimization Strategy

### GPU Allocation Strategy
- **GPU 0**: Agent models (training) - now with chunking
- **GPU 1**: Inference models (target + guard)

### Chunking Hierarchy
1. **Actor Chunking**: Process batches in chunks of 1-2 samples
2. **Critic Chunking**: Process batches in chunks of 1 sample  
3. **High-level Policy Chunking**: Process batches in chunks of 2-4 samples

### Memory Calculation
```python
# Calculate optimal chunk size based on available memory
available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
free_memory_gb = available_memory_gb - allocated_memory_gb

# Conservative memory fractions
actor_memory_fraction = 0.5      # Use 50% of free memory
critic_memory_fraction = 0.4     # Use 40% of free memory

chunk_size = get_optimal_chunk_size(batch_size, free_memory_gb * memory_fraction, seq_len)
chunk_size = max(1, min(chunk_size, max_chunk_size))
```

## Usage

### Run with Optimizations
```bash
# Use the optimized run script
python run_with_memory_optimizations.py

# Or set environment variables manually
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.main
```

### Test Optimizations
```bash
# Test that optimizations work correctly
python test_memory_optimizations.py
```

### Monitor Memory Usage
The system will automatically log memory usage:
```
Using chunked actor training: batch_size=8, seq_len=407, chunk_size=2
Memory stats: available=93.10GB, allocated=15.31GB, free=77.79GB
```

## Configuration Tuning

### If Still Getting OOM
1. **Reduce chunk sizes further:**
   ```json
   "actor_max_chunk_size": 1,
   "critic_max_chunk_size": 1,
   "chunk_size": 1
   ```

2. **Reduce memory fractions:**
   ```json
   "actor_memory_fraction": 0.3,
   "critic_memory_fraction": 0.3,
   "max_gpu_memory": 0.2
   ```

3. **Enable gradient checkpointing:**
   ```json
   "gradient_checkpointing": true
   ```

4. **Reduce LoRA layers:**
   ```json
   "target_layers": [0, 1]  // Further reduction
   ```

### If Memory is Underutilized
1. **Increase chunk sizes:**
   ```json
   "actor_max_chunk_size": 4,
   "critic_max_chunk_size": 2
   ```

2. **Increase memory fractions:**
   ```json
   "actor_memory_fraction": 0.7,
   "max_gpu_memory": 0.35
   ```

## Performance Impact

### Expected Changes
- **Training Speed**: 10-20% slower due to chunking overhead
- **Memory Usage**: 50-70% reduction in peak memory usage
- **Gradient Quality**: Identical (proper gradient accumulation)
- **Model Performance**: No impact on final model quality

### Monitoring
- Watch for `Using chunked actor training` messages
- Monitor `Memory stats` logs
- Check that chunk sizes are reasonable (1-4 samples)

## Troubleshooting

### Still Getting OOM
1. Check if chunking is enabled: Look for "Using chunked actor training" message
2. Verify configuration: Run `test_memory_optimizations.py`
3. Reduce chunk sizes to 1
4. Enable CPU offloading: `"offload_optimizer_to_cpu": true`

### Performance Too Slow
1. Increase chunk sizes gradually
2. Disable chunking for small batches: Remove `actor_chunking_enabled: true`
3. Increase memory fractions if you have spare GPU memory

### Gradient Issues
1. Check that gradient accumulation is working correctly
2. Monitor loss curves for instability
3. Adjust learning rates if needed due to effective batch size changes 