# Code Simplification Guide

## Instruction

Remove redundant type checks (`hasattr`, `isinstance`, `is not None`) and directly access attributes when the structure is deterministic.

## Motivation

Following the "fail fast and loud" principle, we should not hide edge cases behind defensive checks; if the code structure is known and controlled, access attributes directly and let failures surface immediately.

## Example

**Before:**
```python
if hasattr(config, 'training') and config.training is not None:
    debug_mode = getattr(config.training, 'debug_mode', False)
else:
    debug_mode = False

if isinstance(conv_history[0], dict):
    conv_first_msg = conv_history[0].get('content', '')[:100]
else:
    conv_first_msg = str(conv_history[0])[:100]
```

**After:**
```python
debug_mode = config.training.debug_mode

conv_first_msg = conv_history[0]['content'][:100]
```

