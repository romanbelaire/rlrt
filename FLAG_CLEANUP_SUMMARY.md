# Flag Cleanup Summary

## Overview

This document summarizes the cleanup of deprecated command line flags from the hierarchical RL framework codebase.

## Deprecated Flags Removed

### Command Line Flags

1. **`--train`** - Training mode flag
   - **Status**: Removed (no longer used)
   - **Reason**: Replaced by dedicated training flow in main.py
   - **Files updated**: All documentation files

2. **`--test`** - Testing mode flag  
   - **Status**: Removed (no longer used)
   - **Reason**: Replaced by dedicated test_runner.py module
   - **Files updated**: All documentation files

## Code Changes Made

### Source Code Files Updated

1. **`src/main.py`**
   - Removed `--train` and `--test` argument definitions
   - Removed `test_agent()` function (no longer needed)
   - Simplified main() function to only handle training
   - Updated argument parser to remove mode selection group

### Documentation Files Updated

- **`src/README.md`** - Updated all command examples
- **`docs/CONFIGURATION.md`** - Updated configuration examples
- **`docs/CHECKPOINT_LOADING.md`** - Updated checkpoint loading examples
- **`docs/HIERARCHICAL_RL_FRAMEWORK.md`** - Updated framework examples
- **`docs/LOGGING_IMPROVEMENTS.md`** - Updated logging examples
- **`README.md`** - Updated main documentation

## New Command Structure

### Training
```bash
# Old format (deprecated)
python src/main.py --train --config_path config.json

# New format
python src/main.py --config_path config.json
```

### Testing/Evaluation
```bash
# Old format (deprecated)
python src/main.py --test --config_path config.json

# New format
python -m src.test_runner --config-path config.json --checkpoint-dir path/to/checkpoint
```

## Benefits of Flag Cleanup

1. **Simplified Interface**: Removed confusing mode selection
2. **Clear Separation**: Training and testing are now separate modules
3. **Better Architecture**: Dedicated test_runner.py for evaluation
4. **Reduced Complexity**: Fewer command line arguments to manage
5. **Improved Documentation**: Clear examples for each use case

## Current Command Structure

### Training Commands
```bash
# Basic training
python src/main.py --config_path src/configs/config.json

# Training with overrides
python src/main.py --config_path src/configs/config.json --learning_rate 5e-5 --batch_size 16

# Training with checkpoint loading
python src/main.py --config_path src/configs/config.json --load_from_checkpoint path/to/checkpoint
```

### Evaluation Commands
```bash
# Basic evaluation
python -m src.test_runner --config-path config.json --checkpoint-dir path/to/checkpoint

# Evaluation with custom parameters
python -m src.test_runner --config-path config.json --checkpoint-dir path/to/checkpoint --eval-batch-size 8
```

## Migration Guide

### For Users

**Old commands (no longer work):**
```bash
python src/main.py --train --config_path config.json
python src/main.py --test --config_path config.json
```

**New commands:**
```bash
# For training
python src/main.py --config_path config.json

# For evaluation
python -m src.test_runner --config-path config.json --checkpoint-dir path/to/checkpoint
```

### For Developers

When adding new functionality:
1. Use `main.py` for training operations only
2. Use `test_runner.py` for evaluation operations
3. Keep command line interfaces simple and focused
4. Update documentation for all new commands

## Verification

All deprecated flags have been successfully removed from:
- ✅ Source code (no remaining references)
- ✅ Documentation (all examples updated)
- ✅ Command line interface (simplified)
- ✅ Argument parsing (removed mode selection)

The codebase now has a clean, focused command structure with clear separation between training and evaluation workflows. 