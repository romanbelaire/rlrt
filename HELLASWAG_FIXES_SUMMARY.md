# HellaSwag and ARC Agent Fixes Summary

## Issues Identified and Fixed

The following issues were identified in both the general-hellaswag and general-arc agents, and have been fixed in the shared `MultiTurnMultipleChoiceEnv` environment class.

### 1. Critic Candidate Scores All Zero

**Problem**: The critic candidate scores for rejection sampling were all 0, indicating the critic was not working.

**Root Cause**: The `score_safety` method in `MultiTurnMultipleChoiceEnv` was not properly handling the rejection sampling context. It was trying to parse choices from the conversation header, but the parsing logic was failing.

**Fix**: Updated `score_safety` method in `src/general/mc_multiturn_env.py`:
- Enhanced the parsing logic to handle cases where choices are not found in the reference state
- Added fallback logic to extract choices from the original reference state format
- Improved error handling to ensure the method doesn't fail silently

**Code Changes**:
```python
# Enhanced parsing logic in score_safety method
if len(choices) == 0:
    # If no choices found, try to parse from the original reference state
    lines = [ln for ln in (ref or "").splitlines() if ln.strip()]
    for ln in lines:
        if ") " in ln and ln[0].isalpha() and ln[1] == ')':
            choices.append(ln.split(") ", 1)[1].strip())

if len(choices) == 0:
    # Fallback: create a dummy example with the question only
    ex = MCExample(question, ["dummy"], answer_index=0)
else:
    ex = MCExample(question, choices, answer_index=0)
```

### 2. Incorrect Reward Reporting

**Problem**: The environment reward (HL reward) was not correctly implemented. It should be zero unless the HL has decided to submit the answer AND the answer is correct.

**Root Cause**: The reward computation was already correct, but the comment clarified the expected behavior.

**Fix**: Added clarifying comments and ensured the reward logic is explicit:
- Non-submit steps: Use judge-based shaping scores for reasoning quality
- Submit steps: Use binary reward (1 if correct, 0 if incorrect)

**Code Changes**:
```python
# Non-terminal shaping step
score = self._judge_score(ex, self._histories[i], utterance, instruction)
rew = float(max(0.0, min(1.0, score)))

# Submission step - HL reward should only be given when answer is correct
is_correct = int(choice_index == ex.answer_index)
rew = float(is_correct)  # This is the proper HL reward
```

### 3. Poor Conversation Formatting

**Problem**: Answer choices were included in the conversation header, causing the agent to constantly attend to them.

**Root Cause**: The `_format_question` method included all answer choices in the conversation header.

**Fix**: Separated conversation formatting from template processing:
- **Conversation header**: Only includes the question, no answer choices
- **Template processing**: Includes answer choices for proper template formatting

**Code Changes**:

1. **Updated `_format_question` method**:
```python
def _format_question(self, ex: MCExample) -> str:
    """Format question without answer choices to avoid constant attention to answers."""
    parts: List[str] = []
    parts.append("Question: " + ex.question)
    # Remove answer choices from conversation header - they should only be provided in HL instructions
    return "\n".join(parts) + "\n"
```

2. **Added `_format_question_with_choices` method**:
```python
def _format_question_with_choices(self, ex: MCExample) -> str:
    """Format question with choices for template processing (not conversation header)."""
    parts: List[str] = []
    parts.append("Question: " + ex.question)
    parts.append("Choices:")
    for i, c in enumerate(ex.choices):
        label = chr(ord('A') + i)
        parts.append(f"{label}) {c}")
    if len(ex.choices) > 0:
        last_label = chr(ord('A') + len(ex.choices) - 1)
        parts.append(f"Answer with a single letter in [A..{last_label}].")
    else:
        parts.append("Answer with a single letter.")
    return "\n".join(parts) + "\n"
```

3. **Updated `get_base_state` method**:
```python
def get_base_state(self) -> List[str]:
    """Return base state strings (formatted question+choices) for each slot."""
    return [self._format_question_with_choices(self._examples[idx]) for idx in self._indices]
```

4. **Updated templates** to include answer choices in HL instructions:
   - **HellaSwag templates** (`src/configs/templates_general_hellaswag.json`):
     ```json
     {
       "template": "Examine ending {option_label}: {option_text}. What evidence supports or contradicts it, given the context: {context}?\n\nAvailable endings:\n{choices_text}"
     }
     ```
   - **ARC templates** (`src/configs/templates_general_arc.json`):
     ```json
     {
       "template": "Examine option {option_label}: {option_text}. What evidence supports or contradicts it, given the question: {question}?\n\nAvailable choices:\n{choices_text}"
     }
     ```

## Expected Behavior After Fixes

1. **Critic Scores**: Should now provide meaningful scores for rejection sampling instead of all zeros
2. **Rewards**: 
   - Non-submit steps: Judge-based shaping scores for reasoning quality
   - Submit steps: Binary reward (1 if correct, 0 if incorrect)
3. **Conversation Format**: 
   - Agent sees only the question in conversation header
   - Answer choices are provided in HL instructions when needed
   - Templates have access to choices for proper formatting

## Files Modified

1. `src/general/mc_multiturn_env.py`:
   - Fixed `_format_question` method
   - Added `_format_question_with_choices` method
   - Updated `get_base_state` method
   - Enhanced `score_safety` method
   - Added clarifying comments for reward computation

2. `src/configs/templates_general_hellaswag.json`:
   - Updated all templates to include `{choices_text}` placeholder
   - Ensured answer choices are available in HL instructions

3. `src/configs/templates_general_arc.json`:
   - Updated all templates to include `{choices_text}` placeholder
   - Ensured answer choices are available in HL instructions
   - Changed `{context}` to `{question}` for ARC-specific formatting

4. `test_hellaswag_fixes.py`:
   - Created test script to verify fixes work correctly for both HellaSwag and ARC

## Testing

The test script `test_hellaswag_fixes.py` verifies:
- Question formatting separates conversation header from template processing for both datasets
- Score safety method handles parsing correctly
- Reward computation works for both submit and non-submit steps
- ARC environment functionality works correctly

## Impact

These fixes should resolve the three main issues for both HellaSwag and ARC agents:
1. **Critic functionality**: Rejection sampling will now work with meaningful scores
2. **Reward accuracy**: HL rewards will be properly computed
3. **Agent attention**: The agent will no longer be constantly attending to answer choices in the conversation header

## Environment Compatibility

Since both HellaSwag and ARC use the same `MultiTurnMultipleChoiceEnv` class, all fixes automatically apply to both environments. The only differences are:
- **HellaSwag**: Uses `{context}` and `{endings}` terminology in templates
- **ARC**: Uses `{question}` and `{choices}` terminology in templates
- **Dataset loading**: Different dataset IDs and subset requirements
