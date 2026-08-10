# Twenty Questions (20Q) Environment - Complete Guide

This document provides a comprehensive guide to the 20 Questions environment implementation, including how it works, reward structure, win detection, and integration with the trainer. The environment uses a **conversation-based architecture** that leverages the codebase's Conversation infrastructure for proper parallel game management and state tracking.

## How the 20 Questions Task Works

### Game Mechanics

The 20 Questions task is a natural language reasoning game where:

1. **Objective**: Guess a secret word within 20 yes/no questions
2. **Agent Role**: Asks strategic questions to narrow down possibilities
3. **Oracle Role**: Answers questions with "Yes" or "No" based on the secret word
4. **Win Condition**: Correctly guess the word before reaching 20 questions
5. **Win Detection**: Agent must correctly guess the word within 20 questions

### Episode Structure

Each episode follows this structure:

```
Episode Start → Question 1 → Oracle Answer → Question 2 → Oracle Answer → ... → Question 20 → Win/Loss
```

**Key Points:**
- Episodes can end early if the agent correctly guesses the word
- Episodes always end after exactly 20 questions if not won earlier
- No hints are provided - the agent must rely on strategic questioning

### Reward Structure

The reward system is designed to encourage efficient questioning:

- **Winning Episodes**: `reward = 0.0` (terminal reward for correct guess)
- **Non-winning Steps**: `reward = -1.0` (penalty for each question that doesn't lead to a win)
- **Total Episode Reward**: 
  - Successful episodes: `> -20` (since they end with 0 instead of -1)
  - Failed episodes: `-20` (20 questions × -1 reward each)

**Example Reward Sequences:**
- Episode won in 5 questions: `[-1, -1, -1, -1, 0]` = Total: -4
- Episode won in 20 questions: `[-1, -1, ..., -1, 0]` = Total: -19
- Episode failed: `[-1, -1, ..., -1, -1]` = Total: -20

### Win Detection and Metrics

The system tracks wins through multiple mechanisms:

#### 1. **Environment-Level Win Tracking**
```python
# In TwentyQuestionsPolicyEnvironment.step()
if word_guessed:
    reward = 0.0  # No penalty for correct guess
    self.episode_won = True  # Mark episode as won
else:
    reward = -1.0  # Penalty for each question that doesn't lead to correct guess
```

#### 2. **Trainer-Level Win Detection**
```python
# In MCQAgentTrainer.collect_rollout()
if episode_reward >= 0.0:  # 0.0 or positive reward indicates a win
    episode_wins_this_step += 1
    questions_asked = min(steps_collected + 1, 20)
    episode_steps_for_wins.append(questions_asked)
```

#### 3. **Metrics Tracking**
```python
# In update_submit_metrics_twentyq()
if episode_won and episode_steps > 0:
    metrics["submit_wins"] += 1
    metrics["submit_episode_efficiency"].append(episode_steps)
```

### Win Detection

The system tracks wins through multiple mechanisms:

#### 1. **Direct Word Detection**
```python
# In TwentyQuestionsPolicyEnvironment._check_word_guessed()
action_lower = action.lower()
curr_word_lower = curr_word.lower()
if curr_word_lower in action_lower:
    print(f"[DEBUG] Agent directly stated the word '{curr_word}'!")
    return True
```

#### 2. **Answer-Based Detection**
```python
# Check answer only for positive responses
if answer_lower == "yes.":
    # Analyze question content for word matches
    question_clean = question_lower.replace("is it", "").replace("is this", "").replace("?", "").strip()
    # Check if question contains the target word
```

### Question Extraction

The agent may generate verbose responses, so the environment extracts the actual question:

```python
def _extract_question_from_response(self, response: str) -> str:
    """Extract the actual question from the agent's verbose response."""
    lines = response.split('\n')
    question_lines = [line.strip() for line in lines if line.strip().endswith('?')]
    if question_lines:
        return question_lines[-1]  # Return the last question found
    return response  # Fallback to full response
```

### Oracle Response Parsing

The oracle uses robust parsing to handle various response formats:

```python
def _is_yes_response(text: str) -> bool:
    """Check whether oracle response is affirmative."""
    t = (text or "").strip().lower()
    if not t:
        return False
    first = t.split()[0]
    clean_first = first.strip('.,!?')  # Remove punctuation
    return clean_first in {"yes", "y", "yep", "yeah", "correct", "true"}
```

**Supported Response Formats:**
- Exact matches: "yes", "no", "y", "n"
- With punctuation: "Yes.", "No.", "Y!", "N?"
- Variations: "yep", "yeah", "correct", "true"

### Architecture Overview

The 20 Questions environment uses a **conversation-based architecture** that follows the codebase's established patterns:
- **Conversation objects**: Each parallel game is managed by a Conversation object
- **Parallel game management**: Multiple games can run simultaneously with independent state
- **LLM Oracle**: HuggingFace model (default: Llama-3.1-8B-Instruct) for answering yes/no questions
- **State synchronization**: Conversations are the source of truth for game state
- **Enhanced debugging**: Per-conversation logging and game context access

### Key Changes (Recent Architecture Update)

#### 1. **Conversation-Based Architecture**
- **Before**: Single-conversation TextEnv with manual state management
- **After**: Conversation objects for each parallel game with proper state tracking
- **Benefit**: Consistent with codebase patterns, better parallel processing, enhanced debugging

#### 2. **Parallel Game Management**
- **Before**: Single game per environment instance
- **After**: Multiple independent games running simultaneously
- **Benefit**: Better resource utilization and more realistic training scenarios

#### 3. **Enhanced State Management**
- **Before**: Manual state tracking with global variables
- **After**: Conversation objects as source of truth for game state
- **Benefit**: Proper state synchronization, better debugging, easier extension

#### 4. **LLM Oracle Implementation**
- **Oracle Model**: Uses HuggingFace models (configurable via `oracle_model_name`)
- **Prompt Engineering**: Explicit yes/no instructions to ensure proper responses
- **Generation Parameters**: Optimized for short, deterministic responses
- **Word Handling**: Supports both single words and word lists with variants

### Environment Configuration

#### Basic Configuration
```json
{
  "env_task": "twenty_questions",
  "lmrl_options": {
    "max_conversation_length": 20,
    "oracle_model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "oracle_quantize": false,
    "hf_token": null
  }
}
```

#### Oracle Configuration Options
- `oracle_model_name`: HuggingFace model to use as oracle (default: Llama-3.1-8B-Instruct)
- `oracle_quantize`: Enable quantization for memory efficiency (4-bit or 8-bit)
- `max_conversation_length`: Maximum questions per game (default: 20)
- `hf_token`: HuggingFace token for accessing gated models

#### Conversation Management
- **Batch Size**: Number of parallel conversations (games) to run simultaneously
- **Independent State**: Each conversation maintains its own game state
- **State Persistence**: Game progress is tracked per conversation across steps

### Word List

The environment uses a **word list with variants** to support multiple ways of referring to the same object:

```python
word_list = [
    # Sports equipment
    ["Basketball", "Basket ball", "B-ball"],
    ["Football", "American football", "Pigskin"],
    ["Baseball", "Base ball"],
    ["Soccer ball", "Football", "Soccer"],
    
    # Animals
    ["Cat", "Kitty", "Feline", "House cat"],
    ["Dog", "Canine", "Puppy", "Hound"],
    ["Horse", "Equine", "Steed"],
    ["Cow", "Bovine", "Cattle"],
    
    # Fruits
    ["Apple", "Red apple", "Green apple"],
    ["Banana", "Yellow banana"],
    ["Orange", "Citrus fruit"],
    ["Strawberry", "Strawberries"],
    
    # Vehicles
    ["Car", "Automobile", "Vehicle"],
    ["Truck", "Pickup truck", "Semi truck"],
    ["Motorcycle", "Motorbike", "Bike"],
    ["Boat", "Ship", "Vessel"],
    
    # ... and many more
]
```

**Word Selection**: The environment randomly selects one word from the list for each conversation. If a word has multiple variants, the first variant is used as the primary word for oracle responses, but all variants are considered when checking if the agent guessed correctly.

### Oracle Implementation

#### Prompt Format
The oracle uses an explicit prompt format to ensure proper yes/no responses:

```
You are playing 20 questions. Your word is "Cat".

Answer the following question about the word with ONLY a YES or a NO:

Question: Is it an animal?

IMPORTANT: Respond with exactly one word - either "YES" or "NO". Do not provide any explanation or additional text.

Answer:
```

#### Generation Parameters
- `max_new_tokens`: 3 (for short YES/NO responses)
- `do_sample`: False (deterministic generation)
- `num_beams`: 1 (no beam search for speed)

#### Response Parsing
The oracle robustly parses responses to extract yes/no answers:
1. **Exact matches**: "yes", "no", "y", "n"
2. **Word-level parsing**: Check individual words for yes/no patterns (with punctuation removal)
3. **Partial matching**: Fallback for unclear responses
4. **Conservative default**: "No." for unparseable responses

**Recent Fix**: Enhanced punctuation handling to ensure responses like "Yes." and "No." are correctly parsed as affirmative/negative responses. Removed hint mechanism for cleaner gameplay.

### Conversation-Based Architecture

#### Core Components
The environment uses the codebase's Conversation infrastructure for proper state management:

```python
# Each parallel game is managed by a Conversation object
conversation = Conversation(conversation_id)
conversation.add_user_message(question)      # Agent's question
conversation.add_assistant_message(answer)   # Oracle's response
conversation.set_metadata("game_status", "active")  # Game state tracking
```

#### Game State Per Conversation
Each conversation maintains its own game state:

```python
game_state = {
    "curr_word": "Cat",           # Current word to guess
    "question_count": 5,          # Questions asked so far
    "episode_won": False,         # Whether the game was won
    "questions_asked": [...],     # List of questions asked
    "oracle_answers": [...]       # List of oracle responses
}
```

#### Parallel Processing
Multiple conversations can run simultaneously:

```python
# Environment handles multiple games in parallel
env = TwentyQuestionsPolicyEnvironment(
    oracle=oracle,
    word_list=word_list,
    max_conversation_length=20,
    debug_mode=True
)

# Reset with batch size (number of parallel games)
state = env.reset(batch_size=4)  # 4 parallel games

# Step processes all games simultaneously
actions = ["Is it an animal?", "Is it bigger than a car?", "Is it red?", "Is it made of metal?"]
next_state, responses, rewards, dones = env.step(actions)
```

### Integration with Trainer

#### Environment Detection
```python
from src.general.twenty_questions import is_twenty_questions_env

if is_twenty_questions_env(environment):
    # Use 20Q-specific logic
```

#### System Prompt Retrieval
```python
from src.general.twenty_questions import get_system_prompt_twentyq

system_prompt = get_system_prompt_twentyq(environment)
```

#### Instruction Building
```python
from src.general.twenty_questions import make_instruction_from_history_twentyq

instr = make_instruction_from_history_twentyq(state_history: List[str])
```

#### Chat Context Construction
```python
from src.general.twenty_questions import build_chat_contexts_twentyq

contexts = build_chat_contexts_twentyq(
    environment,
    unfinished_patched,
    unfinished_indices,
    send_instructions,
    template_ids_full,
    config,
)
```

#### Metrics Tracking
```python
from src.general.twenty_questions import update_submit_metrics_twentyq

update_submit_metrics_twentyq(metrics, actions, responses)
```

### Debugging and Monitoring

#### Debug Logging
The environment provides comprehensive debug logging when `debug_mode=True`:

```
[LMRL FACTORY] Loading HuggingFace oracle model: meta-llama/Llama-3.1-8B-Instruct
[DEBUG] Initialized conversation 0 with word: '['Cat', 'Kitty', 'Feline']'
[DEBUG] Initialized conversation 1 with word: '['Dog', 'Canine', 'Puppy']'
[ORACLE] Prompt: You are playing 20 questions. Your word is "Cat".
[ORACLE] Question: 'Is it an animal?'
[ORACLE] Raw response: 'yes'
[ORACLE] Parsed answer: 'Yes.'
[DEBUG] Conversation 0: Agent correctly guessed 'Cat' in 5 questions!
[DEBUG] Conversation 1: Game over after 20 questions. Word was 'Dog'
```

#### Key Metrics
- `submit_total_submissions`: Total questions asked
- `submit_valid_submissions`: Valid yes/no questions
- `submit_correct_submissions`: Questions answered "Yes" for correct guesses
- `submit_accuracy_pct`: Percentage of correct guesses
- `submit_avg_steps_to_correct`: Average questions needed for correct guess

### Performance Considerations

#### Memory Optimization
- **Oracle Quantization**: Use `oracle_quantize: 4` for 4-bit quantization
- **Batch Processing**: Environment supports batched interactions
- **GPU Placement**: Oracle can be placed on separate GPU for parallel processing

#### Speed Optimization
- **Deterministic Generation**: `do_sample=False` for faster responses
- **Short Responses**: `max_new_tokens=3` limits generation time
- **No Beam Search**: `num_beams=1` for maximum speed

### Recent Fixes and Improvements

#### Rollout Completion Fix for End Rewards (Latest)
**Problem**: Rollouts would terminate immediately upon reaching the target step count, truncating ongoing conversations and losing valuable end-of-episode rewards (wins in 20Q).

**Root Cause**: 
1. MCQ trainer used premature termination logic: `while steps_collected < rollout_steps`
2. Ongoing conversations that could finish with terminal rewards were cut off
3. Training data lacked important win signals that indicate successful task completion

**Solution**:
- Implemented two-phase rollout termination logic
- Phase 1: Continue collecting until `steps_collected >= rollout_steps` (mark target reached)
- Phase 2: Continue until all conversations complete AND target reached
- Preserves end rewards by allowing started episodes to finish naturally
- Now matches base AgentTrainer's behavior of continuing until `all(already_dones)`

**Benefits**:
- **Better reward signal**: No more truncated episodes missing their terminal rewards
- **Consistent training**: All rollouts contain complete episodes with proper end states
- **Improved win detection**: 20Q wins are never lost due to arbitrary step count cutoffs

#### Architecture Migration to Conversation-Based System
**Problem**: The 20 questions environment was using a single-conversation TextEnv interface instead of leveraging the codebase's Conversation infrastructure.

**Root Cause**: 
1. Environment inherited from TextEnv instead of Environment base class
2. Missing `score_safety` abstract method implementation
3. Manual state management instead of using Conversation objects
4. No support for parallel game processing

**Solution**:
- Migrated to conversation-based architecture using Conversation objects
- Implemented required `score_safety` method (returns zeros for 20Q)
- Added proper parallel game management with independent state tracking
- Enhanced debugging with per-conversation logging

#### Win Detection Fix
**Problem**: Agent was correctly guessing words but showing 0 wins in episode summaries.

**Root Cause**: 
1. Hardcoded reward value (0.0) overriding conditional logic
2. Missing `episode_won` parameter in metrics tracking
3. Efficiency calculation using wrong metrics (step indices instead of questions per episode)

**Solution**:
- Fixed hardcoded reward bug - now properly uses conditional logic
- Added real-time win detection based on reward values (0.0 = win, -1.0 = non-win)
- Fixed metrics tracking to include `episode_won` and `episode_steps` parameters
- Corrected efficiency reporting to show questions per won episode instead of step indices
- Removed hint mechanism for cleaner, more challenging gameplay

#### State Management Fix
**Problem**: State format incompatibility between environment and adapter.

**Solution**: 
- Updated environment to return strings instead of Text objects
- Modified adapter to handle conversation-based environment properly
- Fixed word list handling to support both single words and word variants

#### Conversation History Ordering Fix
**Problem**: Agent was receiving conversation history in the wrong order, causing confusion between agent responses and oracle answers.

**Root Cause**: 
1. The `get_conversation_history()` method in `base.py` was interleaving user and assistant messages by index
2. This broke when the system prompt was added as the first assistant message
3. The resulting order was: [agent_question, system_prompt, agent_question, oracle_answer] instead of [system_prompt, agent_question, oracle_answer, agent_question]

**Solution**:
- Modified `get_conversation_history()` to handle the system prompt correctly
- System prompt (assistant[0]) is now added first
- Remaining messages are properly interleaved: user[0], assistant[1], user[1], assistant[2], etc.
- This ensures the agent sees the conversation in the correct order: system prompt → question → answer → question → answer

**Impact**: Agent now properly distinguishes between its own questions and oracle responses, leading to more strategic questioning behavior.

### Troubleshooting

#### Common Issues

1. **All "No" responses**: Check oracle model loading and prompt format
2. **Memory issues**: Enable quantization or use smaller oracle model
3. **Slow responses**: Verify generation parameters and model placement
4. **0 wins despite correct guesses**: This was a win detection issue that has been fixed
5. **Impossible efficiency values**: This was a metrics calculation issue that has been fixed
6. **Accuracy showing 0% despite correct oracle responses**: This was a punctuation parsing issue that has been fixed
7. **Abstract method errors**: Ensure `score_safety` method is implemented (now fixed)
8. **State format errors**: Check that environment returns strings, not Text objects (now fixed)
9. **Parallel processing issues**: Verify batch_size is set correctly for multiple conversations
10. **Missing end rewards**: Rollouts now properly continue until all conversations complete to preserve terminal rewards (fixed)

#### Debug Mode

Enable debug logging to trace the complete flow:

```python
env = build_lmrl_environment(
    task="twenty_questions",
    debug_mode=True,
    # ... other options
)
```

**Debug Output Examples:**
```
[DEBUG] Agent directly stated the word 'Peas'!
[DEBUG] Episode 0 finished with reward: 0.0
[DEBUG] Episode 0 won in 15 questions! Reward: 0.0
[DEBUG] 20 Questions - Step 14: 1 wins this step
[DEBUG] Episode steps for wins: [15]
```

#### Win Detection Verification

To verify wins are being detected correctly:

1. **Check Debug Output**: Look for `[DEBUG] Episode X won in Y questions!`
2. **Check Episode Summary**: Should show `• Wins: N total wins in rollout`
3. **Check Efficiency**: Should show realistic values like `• Efficiency: 20.0 avg questions per win`
4. **Check Rewards**: Winning episodes should have total reward > -20

### Expected Behavior and Metrics

#### Episode Summary Output

After each episode, you should see output like this:

```
EPISODE   1 SUMMARY
================================================================================
Total time: 1020.00s
Rollout Stats:
  • Timesteps: 256
  • Mean HL Reward (all steps): -1.0000
  • Max HL Reward: -1.0000
  • Min HL Reward: -1.0000
  • Std HL Reward: 0.0000
  • Variance HL Reward: 0.0000
  • Submit HL Reward (meaningful): 0.5317 (2178/4096 correct)
  • Max GPU Allocated (GB): 26.12
Submit Stats:
  • Validity: 95.2% (3898/4096)
  • Accuracy: 53.2% (2178 correct)
  • Efficiency: 20.0 avg questions per win (min: 20, max: 20)
  • Wins: 4 total wins in rollout
  • Episode Efficiency: 20.0 avg questions per win (min: 20, max: 20)
  • Oracle Responses: 2178 YES, 1918 NO
```

#### Key Metrics Explained

- **Wins**: Number of episodes where the agent correctly guessed the word
- **Efficiency**: Average number of questions needed per won episode (should be 1-20)
- **Accuracy**: Percentage of questions that received "Yes" responses from the oracle
- **Validity**: Percentage of non-empty questions asked
- **Mean HL Reward**: Average reward across all steps (typically -1.0 for non-winning steps)

#### Debug Output

With debug mode enabled, you'll see detailed logging:

```
[DEBUG] Agent directly stated the word 'Peas'!
[DEBUG] Episode 0 finished with reward: 0.0
[DEBUG] Episode 0 won in 15 questions! Reward: 0.0
[DEBUG] 20 Questions - Step 14: 1 wins this step
[DEBUG] Episode steps for wins: [15]
[DEBUG] HL rewards this step: [-1.0, -1.0, -1.0, 0.0]
```

### Files
- `src/general/twenty_questions.py`: Helper functions for trainer integration
- `src/general/lmrl_factory.py`: Environment factory with conversation-based 20Q implementation
- `src/general/lmrl_adapter.py`: Adapter for integrating with the trainer system
- `src/base.py`: Conversation class and Environment base class

### Notes
- The environment follows the repository's "fail fast and loud" principle
- All helper functions raise clear errors when required hooks are missing
- CSV logging includes scalar metrics only; detailed predictions are for console output
- The conversation-based architecture provides better parallel processing and debugging
- Each conversation maintains independent game state for proper isolation
- The environment integrates seamlessly with the codebase's Conversation infrastructure