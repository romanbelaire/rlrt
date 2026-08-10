# Oracle Response Parsing Fix

## Overview

This document describes a critical fix for the Twenty Questions environment where oracle responses were not being correctly parsed, leading to inaccurate evaluation metrics.

## Problem Description

### Issue
The oracle was correctly answering questions with responses like `"Yes."` and `"No."`, but the evaluation system was showing 0% accuracy even when the oracle was answering correctly.

### Root Cause
The issue was in the response parsing logic in two places:

1. **Trainer Response Extraction** (`src/mcq_agent_trainer.py`):
   - Oracle responses like `"Yes."` were being extracted as `"yes."` (with period)
   - The period was not being cleaned before passing to the metrics function

2. **Yes/No Detection** (`src/general/twenty_questions.py`):
   - The `_is_yes_response()` function was not handling punctuation
   - `"yes."` was not matching the expected set `{"yes", "y", "yep", "yeah", "correct", "true"}`

### Impact
- Evaluation accuracy showed 0% even when oracle was working correctly
- Training metrics were misleading
- Debugging was difficult because the issue was in parsing, not oracle logic

## Solution

### 1. Enhanced Response Extraction

**File**: `src/mcq_agent_trainer.py` (lines 745-752)

**Before**:
```python
if resp_str.startswith(("yes", "no", "y", "n")):
    first_word = resp_str.split()[0]
    oracle_responses.append(first_word)  # Could include punctuation
```

**After**:
```python
if resp_str.startswith(("yes", "no", "y", "n")):
    first_word = resp_str.split()[0]
    # Clean the first word to remove punctuation
    clean_word = first_word.strip('.,!?')
    oracle_responses.append(clean_word)
```

### 2. Enhanced Yes/No Detection

**File**: `src/general/twenty_questions.py` (lines 22-31)

**Before**:
```python
def _is_yes_response(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    first = t.split()[0]
    return first in {"yes", "y", "yep", "yeah", "correct", "true"}
```

**After**:
```python
def _is_yes_response(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    first = t.split()[0]
    # Clean the first word to remove punctuation
    clean_first = first.strip('.,!?')
    return clean_first in {"yes", "y", "yep", "yeah", "correct", "true"}
```

## Testing

### Before Fix
```
[ENV STEP] Oracle answer: 'Yes.'
[ORACLE] Returning 3 answers: Yes....
...
Label: NO  # Incorrect - should be YES
```

### After Fix
```
[ENV STEP] Oracle answer: 'Yes.'
[ORACLE] Returning 3 answers: Yes....
...
Label: YES  # Correct!
```

### Verification
- Oracle responses like `"Yes."` now correctly parse as `"yes"` → `True` → `"YES"`
- Oracle responses like `"No."` now correctly parse as `"no"` → `False` → `"NO"`
- Evaluation accuracy now shows correct percentages (e.g., 24.8% instead of 0%)

## Files Modified

1. **`src/mcq_agent_trainer.py`**
   - Enhanced response extraction to clean punctuation
   - Ensures clean oracle responses are passed to metrics

2. **`src/general/twenty_questions.py`**
   - Enhanced `_is_yes_response()` function to handle punctuation
   - Ensures proper yes/no detection regardless of punctuation

## Impact

### Positive Changes
- ✅ Accurate evaluation metrics
- ✅ Proper oracle response recognition
- ✅ Correct training feedback
- ✅ Easier debugging and monitoring

### No Breaking Changes
- ✅ Backward compatible with existing responses
- ✅ No changes to oracle behavior
- ✅ No changes to environment interface

## Prevention

### Code Review Guidelines
When working with oracle responses:
1. Always test with responses that include punctuation
2. Verify that `_is_yes_response()` handles edge cases
3. Check that response extraction cleans input properly

### Testing Checklist
- [ ] Test with `"Yes."` responses
- [ ] Test with `"No."` responses  
- [ ] Test with `"Yes!"` responses
- [ ] Test with `"No?"` responses
- [ ] Verify evaluation accuracy is non-zero when oracle answers correctly

## Related Documentation

- `docs/20_questions.md`: General 20Q environment documentation
- `docs/TWENTY_QUESTIONS_SIMPLIFICATION.md`: Architecture simplification details
- `docs/single_agent_archer_twenty_questions_experiment.md`: Experiment-specific documentation

## Future Considerations

### Potential Improvements
1. **More Robust Parsing**: Handle more edge cases (e.g., "YES!!", "no...")
2. **Unicode Support**: Handle different types of punctuation marks
3. **Case Sensitivity**: Ensure consistent handling of mixed case responses

### Monitoring
- Monitor evaluation accuracy to catch similar issues early
- Use debug mode to trace oracle response parsing
- Regular testing with various response formats
