# Hugging Face Dataset Support

This codebase now supports loading datasets directly from Hugging Face, including the `wildjailbreak` and `jailbreakbench` datasets.

## Supported Datasets

The following Hugging Face datasets are currently supported:

- **wildjailbreak**: A dataset of jailbreak prompts for testing model safety
- **jailbreakbench**: A comprehensive benchmark for jailbreak attacks

## Installation

To use Hugging Face datasets, install the required package:

```bash
pip install datasets
```

## Usage

### Configuration

To use a Hugging Face dataset, simply specify the dataset name in your configuration file:

```json
{
  "environment": {
    "data_path": "wildjailbreak",
    // ... other settings
  }
}
```

### Example Configurations

Two example configuration files are provided:

- `src/configs/wildjailbreak_config.json` - For training with the wildjailbreak dataset
- `src/configs/jailbreakbench_config.json` - For training with the jailbreakbench dataset

### Running Training

```bash
# Train with wildjailbreak dataset
python src/main.py --config src/configs/wildjailbreak_config.json

# Train with jailbreakbench dataset
python src/main.py --config src/configs/jailbreakbench_config.json
```

## Dataset Structure

The system automatically detects and handles different dataset structures:

### WildJailBreak Dataset
- **Primary columns**: `prompt`, `question`, `text`
- **Fallback**: Any column containing prompt-like content

### JailBreakBench Dataset
- **Primary columns**: `prompt`, `question`, `text`
- **Fallback**: Any column containing prompt-like content

### Generic Datasets
For other Hugging Face datasets, the system will:
1. Look for a column matching the `question_key` parameter (default: `question`)
2. Fall back to common column names: `prompt`, `question`, `text`, `input`, `content`
3. Use the first suitable column found

## Split Selection

The system automatically selects the best available split in this order:
1. `train` (preferred for training)
2. `test` (fallback)
3. `validation` (fallback)
4. First available split (if none of the above exist)

## Error Handling

If the Hugging Face datasets package is not installed, the system will:
1. Print a warning message
2. Fall back to local file loading
3. Continue with existing functionality

## Testing

A test script is provided to verify Hugging Face dataset support:

```bash
python test_hf_datasets.py
```

This script will:
1. Check if the `datasets` package is available
2. Test dataset detection logic
3. Attempt to load sample datasets (requires internet connection)

## Adding New Datasets

To add support for new Hugging Face datasets:

1. Add the dataset name to the `hf_datasets` list in the `_is_hf_dataset` method
2. Add specific handling logic in the `_load_hf_dataset` method if needed
3. The generic fallback should handle most standard datasets automatically

## Compatibility

- **Backward Compatible**: Existing local file and directory loading continues to work unchanged
- **Graceful Degradation**: If Hugging Face datasets are not available, the system falls back to local loading
- **Flexible**: Supports both local files and Hugging Face datasets seamlessly

## Notes

- Internet connection is required to download Hugging Face datasets on first use
- Datasets are cached locally after first download
- The system automatically handles dataset splits and column detection
- All existing functionality (HarmBench, local files, etc.) remains unchanged 