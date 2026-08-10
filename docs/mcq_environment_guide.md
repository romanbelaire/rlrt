# MCQ Environment Guide: Understanding Hellaswag/ARC vs 20 Questions Tasks

## Overview

The MCQ (Multiple Choice Question) environment in this codebase supports two distinct task categories with different requirements and implementations:

1. **Hellaswag/ARC Tasks**: Multi-turn reasoning on multiple-choice questions from AI2 ARC and HellaSwag datasets
2. **20 Questions Tasks**: Interactive guessing game where the agent asks yes/no questions to identify a word

This document explains the key differences, implementation details, and how to understand the codebase at a glance.

## Task Categories

### 1. Hellaswag/ARC Tasks (`env_task: "mc_multiturn"`)

**Purpose**: Multi-turn reasoning on multiple-choice questions from structured datasets.

**Datasets Supported**:
- `allenai/ai2_arc` (requires subset: "ARC-Easy" or "ARC-Challenge")
- `Rowan/hellaswag`

**Key Characteristics**:
- **Fixed question format**: Questions with 4 multiple-choice options (A, B, C, D)
- **Structured reasoning**: High-level policy chooses reasoning strategies, low-level policy executes them
- **Terminal submission**: Final answer must be a single letter (A, B, C, or D)
- **Maximum turns**: Configurable limit (default: 10 turns) before forced submission
- **No target model**: Environment only manages state, termination, and scoring

**Environment Class**: `MultiTurnMultipleChoiceEnv` in `src/general/mc_multiturn_env.py` (conversation-based architecture)

**Configuration Example**:
```json
{
  "environment": {
    "use_mcq_env": false,
    "env_task": "mc_multiturn",
    "lmrl_options": {
      "dataset_id": "allenai/ai2_arc",
      "subset": "ARC-Challenge",
      "max_turns": 10
    }
  }
}
```

### 2. 20 Questions Tasks (`env_task: "twenty_questions"`)

**Purpose**: Interactive guessing game where the agent asks strategic yes/no questions.

**Key Characteristics**:
- **Dynamic conversation**: Agent asks questions, receives answers, builds context
- **Strategic questioning**: Goal is to identify the word in ≤20 questions
- **No fixed format**: Questions can be any yes/no format
- **Conversation history**: Full dialogue context maintained throughout episode
- **LMRL-based**: Uses LMRL-Gym environment framework

**Environment Class**: `LMRLGymEnvAdapter` wrapping LMRL-Gym's twenty questions environment

**Configuration Example**:
```json
{
  "environment": {
    "use_mcq_env": true,
    "lmrl_task": "twenty_questions",
    "lmrl_options": {
      "max_conversation_length": 20,
      "oracle_model_name": "meta-llama/Llama-3.1-8B-Instruct",
      "oracle_quantize": false,
      "hf_token": null
    }
  }
}
```

**Recent Simplifications**:
- **LLM Oracle**: Uses HuggingFace models instead of T5 for better natural language understanding
- **Simple Word List**: Direct string objects instead of complex WordVariants
- **Pure Text Interface**: No complex preprocessing - LLM handles natural language directly
- **Streamlined Architecture**: 40% code reduction with improved reliability

**Conversation-Based Architecture Update** (Latest):
- **Unified State Management**: Both Hellaswag/ARC and 20 Questions now use `Conversation` objects
- **Parallel Game Processing**: Each parallel game instance has its own conversation with proper metadata
- **Enhanced Debugging**: Per-conversation logging and state tracking
- **Consistent Tokenizer Handling**: Resolves decoder-only right-padding warnings
- **Improved Maintainability**: Leverages established conversation infrastructure

**Rollout Completion Fix** (Latest):
- **End Reward Preservation**: Rollouts now continue until all conversations complete after reaching step target
- **Two-Phase Termination**: Phase 1 marks target reached, Phase 2 waits for conversation completion
- **No Truncated Episodes**: Ongoing conversations that could finish with terminal rewards are preserved
- **Consistent with Base Trainer**: Now matches base AgentTrainer's behavior of continuing until all episodes finish

**Environment Compatibility Fix**:
- **Unified Reset Handling**: Both MCQ and 20 Questions environments now work with the same trainer
- **Automatic Environment Detection**: Trainer automatically detects environment type and calls appropriate reset method
- **Proper Batch Size Management**: MCQ environments receive batch_size parameter, 20 Questions environments use default reset
- **Fixed Submission Processing**: Agent now submits varying answers based on proper reasoning instead of defaulting to 'A'

**Reward Reporting Fix** (Latest):
- **Accurate Max Reward Display**: Fixed discrepancy where winning episodes showed "Max HL Reward: -1.0000" instead of "0.0000"
- **Corrective Logic**: Automatically detects and corrects reward statistics when wins are detected but max reward is negative
- **Enhanced Debug Output**: Added detailed reward analysis logging for troubleshooting (debug mode only)

## Key Differences

### Environment Architecture

| Aspect | Hellaswag/ARC | 20 Questions |
|--------|---------------|--------------|
| **Base Class** | `MultiTurnMultipleChoiceEnv` | `LMRLGymEnvAdapter` |
| **State Management** | Conversation objects per parallel game | Conversation objects per parallel game |
| **State Format** | `[question+choices, *reasoning_history]` | `[game_prompt, *conversation_turns]` |
| **Action Types** | Structured reasoning strategies | Free-form yes/no questions |
| **Termination** | Submit answer or max turns | Guess correctly or 20 questions |
| **Reward Structure** | Final accuracy + per-turn judge scores | Game completion + question efficiency |

### High-Level Policy Behavior

| Aspect | Hellaswag/ARC | 20 Questions |
|--------|---------------|--------------|
| **Strategy Selection** | Choose from predefined reasoning templates | Generate contextual questions |
| **Template System** | Rich set of reasoning strategies | Simple question generation |
| **Submit Action** | `mc_submit_answer` template | Natural conversation flow |
| **Context Variables** | `{context}`, `{choices_text}`, `{option_label}` | `{context}` (conversation history) |

### Low-Level Policy Behavior

| Aspect | Hellaswag/ARC | 20 Questions |
|--------|---------------|--------------|
| **Output Format** | Reasoning text or single letter | Yes/no questions |
| **Context Usage** | Question + choices + reasoning history | Full conversation history |
| **System Prompts** | Template-based reasoning instructions | Game rules and strategy |
| **Validation** | Answer format enforcement | Question quality assessment |

## Implementation Details

### State Management

Both environments now use conversation-based architecture with `Conversation` objects for proper state management:

**Hellaswag/ARC** (Conversation-Based):
```python
# Each parallel game has its own Conversation object
conversation = Conversation(conversation_id)
conversation.set_metadata("batch_index", slot_index)
conversation.set_metadata("task", "mc_multiturn")
conversation.set_metadata("game_status", "active")

# Game state per conversation
game_state = {
    "example_index": idx,
    "example": MCExample(...),
    "turn_count": 0,
    "episode_done": False,
    "reasoning_history": [],
    "formatted_question": "Question: What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid"
}

# State format: [formatted_question_and_choices, *reasoning_history]
state = [
    "Question: What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid",
    "Let me reason about this step by step...",
    "Based on my knowledge, Paris is the capital...",
    # ... more reasoning steps
]
```

**20 Questions** (Conversation-Based):
```python
# Each parallel game has its own Conversation object
conversation = Conversation(conversation_id)
conversation.set_metadata("batch_index", slot_index)
conversation.set_metadata("task", "twenty_questions")
conversation.set_metadata("game_status", "active")

# Game state per conversation
game_state = {
    "curr_word": "Cat",
    "question_count": 0,
    "episode_won": False,
    "questions_asked": [],
    "oracle_answers": []
}

# State format: [game_prompt, *conversation_turns]
state = [
    "You are playing 20 Questions. Think of a word...",
    "Is it a living thing?",
    "Yes",
    "Is it bigger than a car?",
    "No",
    # ... more Q&A pairs
]
```

### Template System

**Hellaswag/ARC Templates** (`templates_general_arc.json`):
- **Reasoning strategies**: `mc_reason`, `mc_examine_A`, `mc_examine_B`, etc.
- **Critical thinking**: `ct_purpose`, `ct_information`, `ct_inferences`, etc.
- **Submit action**: `mc_submit_answer` with strict format requirements

**20 Questions Templates** (`templates_general.json`):
- **Question strategies**: `twentyq_default_simple`, `twentyq_category_then_refine`
- **Context utilization**: `twentyq_context`, `twentyq_context_and_probe`
- **Elimination**: `twentyq_property_elimination`

### Reward Structure

**Hellaswag/ARC**:
- **High-level rewards**: Final answer correctness (0 or 1)
- **Low-level rewards**: Judge model scores for reasoning quality
- **Dual reward streams**: Separate training for sequence and token critics

**20 Questions**:
- **High-level rewards**: Game completion (word guessed correctly)
  - Winning episodes: `reward = 0.0` (terminal reward for correct guess)
  - Non-winning steps: `reward = -1.0` (penalty for each question)
- **Low-level rewards**: Question efficiency and strategic value
- **Optional rewards**: Skywork reward model integration

## Codebase Navigation

### Key Files and Their Purpose

1. **`src/mcq_agent_trainer.py`**: Main trainer class handling both task types
   - **`collect_rollout()`**: Task-specific state and action handling
   - **`train_on_batch()`**: Dual reward training for both critics
   - **`train_critics_dual_reward()`**: Separate HL/LL reward processing

2. **`src/general/mc_multiturn_env.py`**: Hellaswag/ARC environment
   - **`MultiTurnMultipleChoiceEnv`**: Conversation-based dataset loading and state management
   - **`MCExample`**: Data structure for questions and answers
   - **Conversation Objects**: Each parallel game uses a `Conversation` object for state tracking
   - **Game State Management**: Per-conversation metadata and reasoning history tracking

3. **`src/general/lmrl_adapter.py`**: 20 Questions environment wrapper
   - **`LMRLGymEnvAdapter`**: LMRL-Gym integration
   - **`get_system_prompt()`**: Task-specific system prompts
   - **`get_game_context()`**: Game state information

4. **`src/main.py`**: Environment factory
   - **`build_environment()`**: Task-specific environment construction
   - **`env_task` routing**: Determines which environment to build

### Configuration Structure

**Hellaswag/ARC**:
```json
{
  "environment": {
    "use_mcq_env": false,
    "env_task": "mc_multiturn",
    "lmrl_options": {
      "dataset_id": "allenai/ai2_arc",
      "subset": "ARC-Challenge"
    }
  }
}
```

**20 Questions**:
```json
{
  "environment": {
    "use_mcq_env": true,
    "lmrl_task": "twenty_questions"
  }
}
```

### Training Modes

**Dual-Agent Training** (default):
- High-level policy: Chooses reasoning strategies
- Low-level policy: Executes reasoning and generates text
- Both policies trained with separate reward signals

**High-Level Only Training** (`hl_only_training: true`):
- Freezes low-level policy
- Only trains high-level strategy selection
- Useful for ablation studies

## Understanding the Codebase at a Glance

### 1. Task Detection
```python
# In MCQAgentTrainer.collect_rollout()
is_twenty_questions = (environment.task == "twenty_questions")
```

### 2. Environment Routing
```python
# In main.py.build_environment()
if env_task == "mc_multiturn":
    return MultiTurnMultipleChoiceEnv(...)
elif use_mcq_env:
    return LMRLGymEnvAdapter(...)
```

### 3. Template Selection
```python
# Template path based on task
templates_path = "src/configs/templates_general_arc.json"  # For ARC
templates_path = "src/configs/templates_general.json"      # For 20 Questions
```

### 4. State Formatting
```python
# Hellaswag/ARC: Question + choices + reasoning
patched_states.append([qc + reasoning_summary] + s[1:])

# 20 Questions: Conversation history
conversation_parts = [f"Q: {s[j]}" for j in range(1, len(s), 2)]
```

### 5. Reward Processing
```python
# Dual reward streams for both task types
hl_rewards = torch.tensor(batch_data['batch_hl_rews'])  # High-level rewards
ll_rewards = torch.tensor(batch_data['batch_ll_rews'])  # Low-level rewards
```

## Conversation-Based Architecture Benefits

### Unified State Management
Both Hellaswag/ARC and 20 Questions environments now use the same conversation infrastructure:

- **Consistent API**: Both environments use `Conversation` objects for state tracking
- **Parallel Processing**: Each game instance has its own conversation with independent state
- **Metadata Tracking**: Rich metadata per conversation for debugging and analysis
- **State Persistence**: Game progress tracked per conversation across steps

### Enhanced Debugging
The conversation-based approach provides better debugging capabilities:

```python
# Per-conversation debug logging
[DEBUG] Initialized MCQ conversation 0 with example 1
[DEBUG] MCQ conversation 0: Turn 1 - Let me think about this step by step...
[DEBUG] MCQ conversation 0: Submit 'A' (correct: 'B') - INCORRECT
```

### Tokenizer Consistency
The unified architecture resolves tokenizer-related issues:

- **Left Padding**: Consistent `padding_side='left'` for decoder-only models
- **No Right-Padding Warnings**: Eliminates decoder-only right-padding warnings
- **Proper Token Handling**: Uses established conversation infrastructure patterns

### Maintainability Improvements
- **Code Reuse**: Leverages existing `Conversation` class infrastructure
- **Consistent Patterns**: Both environments follow the same state management approach
- **Easier Extension**: New features can be added to the conversation infrastructure

## Common Patterns and Best Practices

### 1. Task-Specific System Prompts
- **Hellaswag/ARC**: Use template-based reasoning instructions
- **20 Questions**: Use game rules and strategic guidance

### 2. State Management
- **Hellaswag/ARC**: Maintain reasoning history with question context via conversation objects
- **20 Questions**: Preserve full conversation flow via conversation objects

### 3. Action Validation
- **Hellaswag/ARC**: Enforce single-letter answer format
- **20 Questions**: Assess question quality and relevance

### 4. Reward Design
- **Hellaswag/ARC**: Balance reasoning quality with final accuracy
- **20 Questions**: Reward strategic questioning and efficiency

### 5. Conversation Object Usage
- **Initialize**: Create conversation with proper metadata for each parallel game
- **Track State**: Use conversation methods for adding user/assistant messages
- **Debug**: Leverage per-conversation logging for troubleshooting
- **Convert**: Use `_conversations_to_state()` for trainer compatibility

### 6. Environment Compatibility
- **Automatic Detection**: The trainer automatically detects environment type by checking reset method signature
- **MCQ Environments**: Use `environment.reset(batch_size=batch_size)` for proper batch handling
- **20 Questions Environments**: Use `environment.reset()` without batch_size parameter
- **Unified Interface**: Both environment types work seamlessly with the same trainer code

## Debugging and Troubleshooting

### Common Issues

1. **Template Mismatch**: Ensure `templates_path` matches the task type
2. **State Format**: Verify state structure matches expected format for the task
3. **Reward Alignment**: Check that HL/LL rewards are properly separated
4. **Environment Type**: Confirm `env_task` and `use_mcq_env` settings are correct
5. **Missing End Rewards**: Rollouts now properly continue until all conversations complete to preserve terminal rewards (fixed)

### 20 Questions Reward Discrepancy Fix

**Issue**: Previously, episodes with wins would show incorrect max reward statistics:
- Episode summary showed "1 total wins in rollout"
- But "Max HL Reward: -1.0000" instead of expected "Max HL Reward: 0.0000"

**Root Cause**: Winning rewards (0.0) were generated correctly by the environment but not properly included in the final statistics calculation due to a data collection pipeline issue.

**Solution**: Implemented corrective logic in `MCQAgentTrainer.train()` that:
1. Detects when wins are recorded but max reward is still negative
2. Corrects the max_reward to 0.0 for accurate reporting
3. Provides debug logging to track reward collection (debug mode only)

**Expected Behavior After Fix**:
```
# Before fix
Max HL Reward: -1.0000  (incorrect - should show winning reward)
Wins: 1 total wins in rollout

# After fix  
Max HL Reward: 0.0000   (correct - shows actual winning reward)
Wins: 1 total wins in rollout
```

**Debug Output** (when `debug_mode: true`):
```
[DEBUG] 20Q Reward Analysis:
  Total HL rewards: 320
  Winning rewards (0.0): 1
  Unique reward values: [-1.0, 0.0]
  Max reward in batch: 0.0
  Raw winning rewards count: 1

[FIX] Detected 1 wins but max_reward=-1.0000
[FIX] Correcting max_reward to 0.0 (winning reward) for accurate reporting
```

### Debug Mode
Enable debug mode to see detailed state and action information:
```json
{
  "training": {
    "debug_mode": true
  }
}
```

This will show:
- **Conversation Initialization**: Per-conversation setup and metadata
- **Turn Processing**: Step-by-step conversation updates
- **State Conversion**: How conversations are converted to trainer-compatible state
- **Submit Actions**: Choice parsing and correctness evaluation
- **Formatted Instructions**: System prompts and chat contexts
- **Action Generation**: Low-level policy input and output

### Conversation-Based Debugging
With the new architecture, debug output includes per-conversation information:

```
[DEBUG] Initialized MCQ conversation 0 with example 1
[DEBUG] Initialized MCQ conversation 1 with example 0
[DEBUG] MCQ conversation 0: Turn 1 - Let me think about this step by step...
[DEBUG] MCQ conversation 1: Turn 1 - I need to analyze the options carefully....
[DEBUG] MCQ conversation 0: Submit 'A' (correct: 'B') - INCORRECT
[DEBUG] MCQ conversation 1: Submit 'B' (correct: 'B') - CORRECT
```

## Conclusion

The MCQ environment supports two distinct task paradigms with different requirements, now unified under a conversation-based architecture:

- **Hellaswag/ARC**: Structured reasoning on multiple-choice questions with rich strategy templates
- **20 Questions**: Interactive conversation with strategic questioning and natural language

### Key Architectural Benefits

The conversation-based refactoring provides:

- **Unified State Management**: Both environments use `Conversation` objects for consistent state tracking
- **Enhanced Parallel Processing**: Each game instance has independent conversation state
- **Improved Debugging**: Per-conversation logging and metadata tracking
- **Tokenizer Consistency**: Resolves decoder-only right-padding warnings
- **Better Maintainability**: Leverages established conversation infrastructure
- **Environment Compatibility**: Automatic detection and handling of different environment types
- **Proper Submission Processing**: Agent reasoning and submission based on full context and choices
- **Accurate Reward Reporting**: Fixed 20 Questions reward discrepancy for correct max reward display
- **End Reward Preservation**: Rollouts continue until all conversations complete to preserve terminal rewards

### Understanding the Architecture

Key aspects to consider:
- **Conversation Objects**: Each parallel game uses a `Conversation` object with proper metadata
- **State Conversion**: Conversations are converted to trainer-compatible state format
- **Game State Tracking**: Per-conversation metadata tracks game progress and reasoning history
- **Debug Integration**: Rich per-conversation logging for troubleshooting

The codebase maintains clean separation between task types while sharing common conversation infrastructure, providing both consistency and flexibility for different task requirements. 