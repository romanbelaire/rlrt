# Hierarchical 20 Questions Implementation Guide

## Overview

This document provides a comprehensive guide to the **hierarchical (dual-agent) 20 Questions implementation**, featuring dual-agent training with separate high-level (meta-cognitive) and low-level (question generation) policies.

**Important Distinction**: This is different from the single-agent 20Q implementation in the MCQ trainer, which uses a single policy with conversation history management but no hierarchical structure.

## Architecture

### Core Components

1. **TwentyQuestionsConversation**: Extended conversation class with 20Q-specific metadata
2. **HierarchicalTwentyQuestionsEnvironment**: Environment supporting dual-agent training
3. **HierarchicalTwentyQTrainerMixin**: Trainer extensions for hierarchical support
4. **HuggingFaceOracle**: LLM-based oracle for answering questions
5. **System Prompts**: Specialized prompts for HL and LL agents

### Agent Roles (Dual-Agent Architecture)

This hierarchical implementation uses **two separate agents** working in coordination, unlike the single-agent 20Q in the MCQ trainer.

#### High-Level Agent (Meta-Cognitive)
- **Purpose**: Provides strategic guidance and meta-cognitive reasoning
- **Input**: Conversation history and information state summary
- **Output**: Free-form strategic instructions for low-level agent
- **Rewards**: Game outcome based (-1.0 per step, 0.0 for wins)
- **Training**: Separate policy network with its own optimizer

**Example High-Level Actions**:
- "Ask about the broad category - determine if this is a living thing or not."
- "Focus on size - find out if this object is larger than a car."
- "Make your final guess - based on the clues, I believe the word is 'elephant'."

#### Low-Level Agent (Question Generation)
- **Purpose**: Generates actual yes/no questions based on high-level guidance
- **Input**: High-level instruction + conversation history (from oracle perspective)
- **Output**: Specific yes/no questions or word guesses
- **Rewards**: Instruction-following quality (0.0-1.0 range)
- **Training**: Separate policy network with its own optimizer

**Example Low-Level Actions**:
- "Is it a living thing?" (following category guidance)
- "Is it larger than a car?" (following size guidance)
- "Is it an elephant?" (following guess guidance)

### Comparison with Single-Agent 20Q

| Aspect | Hierarchical 20Q (This Implementation) | Single-Agent 20Q (MCQ Trainer) |
|--------|----------------------------------------|--------------------------------|
| **Architecture** | Dual-agent: separate HL and LL policies | Single-agent: one policy |
| **Training** | Two separate optimizers and reward signals | One optimizer and reward signal |
| **Context Management** | Oracle-perspective unified context | Dialogue-based conversation history |
| **Strategic Planning** | Explicit high-level strategic reasoning | Implicit strategy within single policy |
| **Complexity** | Higher complexity, more parameters | Lower complexity, fewer parameters |
| **Use Case** | Research into hierarchical reasoning | Standard RL for 20Q gameplay |

## File Structure

```
src/
├── hierarchical_conversation.py       # Extended Conversation class
├── hierarchical_20q_prompts.py       # System prompts and formatting
├── hierarchical_twentyq_env.py       # Core environment implementation
├── hierarchical_twentyq_trainer.py   # Trainer extensions
├── hierarchical_twentyq_factory.py   # Environment factory functions
└── main.py                          # Updated with hierarchical 20Q support

configs/
└── hierarchical_20q_example.json     # Example configuration

docs/
├── hierarchical_20q_plan.md          # Original planning document
└── hierarchical_20q_implementation.md # This implementation guide
```

## Configuration

### Basic Configuration

```json
{
  "model": {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "use_high_level_policy": true,
    "high_level_policy_type": "llm"
  },
  "environment": {
    "env_task": "hierarchical_twenty_questions",
    "lmrl_options": {
      "oracle_model_name": "meta-llama/Llama-3.1-8B-Instruct",
      "max_conversation_length": 20,
      "hl_reward_success": 0.0,
      "hl_reward_failure": -1.0,
      "ll_reward_range": [0.0, 1.0]
    }
  },
  "training": {
    "batch_size": 4,
    "hierarchical_20q_mode": "default"
  }
}
```

### Advanced Configuration Options

#### Oracle Configuration
- `oracle_model_name`: HuggingFace model for oracle (default: Llama-3.1-8B-Instruct)
- `oracle_quantize`: Quantization for oracle model (4, 8, or false)
- `hf_token`: HuggingFace authentication token

#### Judge Model Configuration
- `judge_model_name`: Model for evaluating instruction-following quality
- `judge_quantize`: Quantization for judge model

#### Reward Configuration
- `hl_reward_success`: High-level reward for winning episodes (default: 0.0)
- `hl_reward_failure`: High-level reward for non-winning steps (default: -1.0)
- `ll_reward_range`: Range for low-level instruction-following rewards (default: [0.0, 1.0])

## Usage

### Training

```bash
python src/main.py --config configs/hierarchical_20q_example.json
```

### Key Training Parameters

- `batch_size`: Number of parallel games (default: 4)
- `rollout_steps`: Steps to collect per rollout (default: 64)  
- `hierarchical_20q_mode`: Training mode ("default", "hl_only", "ll_only")
- `debug_mode`: Enable detailed logging (recommended for development)

## Implementation Details

### Conversation Management

Each game uses a `TwentyQuestionsConversation` object that tracks:

```python
# Game state
target_word: str                    # Word to guess
question_count: int                 # Questions asked so far
episode_won: bool                   # Whether game was won
episode_done: bool                  # Whether game is complete

# Question tracking
questions_asked: List[str]          # All questions asked
oracle_answers: List[str]           # Oracle responses
question_types: List[str]           # Categories of questions

# Strategy tracking
hl_strategies: List[str]            # High-level instructions
confirmed_properties: List[str]     # Properties confirmed as true
ruled_out_properties: List[str]     # Properties ruled out as false
categories_explored: List[str]      # Question categories used

# Reward tracking
hl_rewards: List[float]             # High-level policy rewards
ll_rewards: List[float]             # Low-level policy rewards
```

### Dual Reward System

#### High-Level Rewards (Game Outcome)
- **Success**: 0.0 (no penalty for correct guess)
- **Failure**: -1.0 (penalty for each non-winning step)
- **Purpose**: Encourages efficient questioning and correct guesses

#### Low-Level Rewards (Instruction Following)
- **Range**: 0.0-1.0 (configurable)
- **Evaluation**: Judge model or heuristic-based
- **Purpose**: Encourages following high-level strategic guidance

### Oracle Implementation

The `HuggingFaceOracle` uses a language model to answer questions:

```python
# Oracle prompt format
prompt = f"""You are playing 20 questions. Your word is "{word}".

Answer the following question about the word with ONLY a YES or a NO:

Question: {question}

IMPORTANT: Respond with exactly one word - either "YES" or "NO".

Answer:"""
```

### System Prompts

#### High-Level Agent System Prompt
- Emphasizes strategic thinking and meta-cognitive reasoning
- Provides clear guidance on available strategic approaches
- Focuses on analyzing conversation state and providing actionable instructions

#### Low-Level Agent System Prompt  
- Emphasizes following strategic guidance from partner
- Focuses on generating clear yes/no questions
- Provides examples of how to interpret high-level instructions

## Training Process

### Rollout Collection

1. **Environment Reset**: Initialize conversations with target words
2. **High-Level Action Generation**: Generate strategic instructions
3. **Low-Level Action Generation**: Generate questions based on instructions
4. **Environment Step**: Execute questions, get oracle responses
5. **Reward Calculation**: Compute dual rewards for both policies
6. **State Update**: Update conversation histories and metadata

### Dual Policy Training

1. **High-Level Policy**: Trained on game outcome rewards using PPO
2. **Low-Level Policy**: Trained on instruction-following rewards using PPO
3. **Critics**: Separate critics for high-level and low-level value estimation
4. **Synchronization**: Both policies trained simultaneously with separate optimizers

## Debugging and Monitoring

### Debug Output

With `debug_mode: true`, you'll see detailed logging:

```
[DEBUG] Initialized hierarchical 20Q conversation 0 with word: 'Cat'
[ORACLE] Word: 'Cat', Question: 'Is it a living thing?'
[ORACLE] Raw response: 'yes'
[ORACLE] Parsed answer: 'Yes.'
[DEBUG] Hierarchical 20Q conversation 0: Won in 8 questions!
```

### Key Metrics

- **Win Rate**: Percentage of games won within 20 questions
- **Efficiency**: Average questions needed per win
- **Instruction Following**: Average low-level reward (0.0-1.0)
- **Strategy Distribution**: Types of high-level strategies used
- **Oracle Response Distribution**: Ratio of Yes/No responses

## Performance Considerations

### Memory Optimization
- Oracle quantization: Use 4-bit or 8-bit quantization for memory efficiency
- Judge model quantization: Optional judge model for instruction evaluation
- Batch processing: Environment supports parallel game processing

### Speed Optimization
- Deterministic generation: `do_sample=False` for faster oracle responses
- Short responses: `max_new_tokens=3` for oracle efficiency
- Caching: Model loading is cached between runs

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all hierarchical 20Q modules are properly imported
2. **Memory Issues**: Enable quantization for oracle and judge models
3. **Slow Training**: Reduce batch size or enable quantization
4. **Low Win Rates**: Check oracle model performance and word list difficulty
5. **Poor Instruction Following**: Verify judge model configuration or heuristic scoring

### Debug Mode

Enable comprehensive debugging:

```json
{
  "training": {
    "debug_mode": true
  }
}
```

This provides:
- Conversation initialization details
- Oracle question-answer pairs
- High-level strategy instructions
- Low-level question generation
- Reward calculations
- Win/loss detection

## Integration with Existing Codebase

### Trainer Integration
- Extends `MCQAgentTrainer` with hierarchical 20Q support
- Uses monkey-patching to add new methods without breaking existing functionality
- Automatic detection of hierarchical 20Q environments
- **Note**: Coexists with single-agent 20Q in MCQ trainer without conflicts

### Environment Factory
- Integrated with main environment factory in `src/main.py`
- Uses `env_task: "hierarchical_twenty_questions"` for detection (distinct from `"twenty_questions"`)
- Compatible with existing configuration system
- **Environment Distinction**:
  - `"twenty_questions"` → Single-agent 20Q (MCQ trainer)
  - `"hierarchical_twenty_questions"` → Dual-agent 20Q (this implementation)

### Conversation Infrastructure
- Extends base `Conversation` class with 20Q-specific metadata
- Maintains compatibility with existing conversation logging
- Leverages established conversation patterns
- **Context Management**: Uses oracle-perspective formatting (different from MCQ trainer's dialogue approach)

## Future Extensions

### Potential Improvements

1. **Advanced High-Level Policies**: Template-based or hybrid approaches
2. **Curriculum Learning**: Progressive difficulty in word selection
3. **Multi-Modal**: Support for image-based 20 Questions
4. **Human Studies**: Evaluation against human performance
5. **Transfer Learning**: Cross-domain strategy transfer

### Research Directions

1. **Meta-Learning**: Learning to learn questioning strategies
2. **Few-Shot Adaptation**: Quick adaptation to new word categories
3. **Explainable AI**: Interpretable strategy selection
4. **Collaborative AI**: Human-AI collaborative questioning

## Conclusion

The hierarchical 20 Questions implementation provides a complete framework for training **dual-agent systems** with meta-cognitive reasoning capabilities. This is distinct from the single-agent 20Q approach in the MCQ trainer, offering a more complex but potentially more powerful architecture for studying hierarchical reasoning.

Key advantages of the dual-agent approach:
- **Explicit Strategic Reasoning**: High-level agent provides interpretable strategic decisions
- **Modular Training**: Separate optimization of strategic and execution components  
- **Research Flexibility**: Enables studying different combinations of HL/LL policies
- **Context Optimization**: Oracle-perspective formatting reduces three-way conversation confusion

The architecture is modular, extensible, and integrates seamlessly with the existing codebase while coexisting with the single-agent 20Q implementation. It provides rich debugging and monitoring capabilities and demonstrates how high-level strategic reasoning can be combined with low-level execution in a reinforcement learning setting, providing a foundation for more sophisticated hierarchical AI systems.
