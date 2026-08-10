# ATV Adaptation for 20 Questions Environment - Design Requirements Document

## Overview

This document outlines the design requirements and implementation plan for adapting the Adaptive Task Vector (ATV) high-level policy from the MCQ (Hellaswag/ARC) environment to work with the 20 Questions environment. The adaptation follows the successful pattern established in `ATVMCQAgentTrainer` but addresses the unique characteristics of the 20 Questions task.

## Background

### Current ATV Implementation (MCQ)
- **Base Class**: `ATVMCQAgentTrainer` extends `MCQAgentTrainer`
- **Key Features**: ATV-based high-level policy with automatic submission action detection
- **Environment**: Hellaswag/ARC multiple-choice questions with structured reasoning
- **Action Types**: Template-based reasoning strategies with final answer submission
- **Reward Structure**: Final accuracy + per-turn judge scores

### Target Environment (20 Questions)
- **Base Class**: Will extend `MCQAgentTrainer` (already supports 20Q)
- **Key Features**: Dynamic conversation with yes/no questions and word guessing. No need for submit action detection, because every LL action is submitted.
- **Environment**: Interactive guessing game with LLM oracle
- **Action Types**: Free-form yes/no questions and word guesses
- **Reward Structure**: Game completion + question efficiency (-1 per question, 0 for wins)

## Key Differences Analysis

### 1. Action Types and Patterns

| Aspect | MCQ (Current ATV) | 20 Questions (Target) |
|--------|-------------------|----------------------|
| **Action Format** | Structured reasoning + `SUBMIT ACTION: [A-D]` | Free-form yes/no questions + word guesses |
| **Detection Pattern** | `SUBMIT_ACTION_PATTERN = r"SUBMIT ACTION\s*:\s*([A-D])"` | No need for submit action detection, because every LL action is submitted. |
| **Template Mapping** | `SUBGOAL_TO_TEMPLATE` mapping for reasoning strategies | Not applicable. |
| **Termination** | Submit final answer choice | Guess the correct word |

Note: Templates are not cleanly recoverable from latent states. in ATV models, template mapping this feature is not applicable.

### 2. Environment Integration

| Aspect | MCQ | 20 Questions |
|--------|-----|--------------|
| **State Format** | `[question+choices, *reasoning_history]` | `[game_prompt, *conversation_turns]` |
| **Environment Class** | `MultiTurnMultipleChoiceEnv` | `LMRLGymEnvAdapter` |
| **Reset Method** | `reset(batch_size=env_batch_size)` | `reset(batch_size=env_batch_size)` (supports batch_size) |
| **Step Method** | `step(actions, template_ids=template_ids)` | `step(actions)` (no template_ids) |

### 3. Reward Structure

| Aspect | MCQ | 20 Questions |
|--------|-----|--------------|
| **High-Level Rewards** | Final answer correctness (0 or 1) | Game completion (0 for wins, -1 per question) |
| **Low-Level Rewards** | Judge model scores for reasoning quality | Question efficiency and strategic value |
| **Terminal Rewards** | Immediate reward on correct submission | Immediate reward on correct word guess |

## Key Insights from Existing Implementation

### Environment Interface
- **Reset Method**: Both MCQ and 20Q environments support `reset(batch_size=env_batch_size)`
- **Step Method**: 20Q environment only accepts `step(actions)` (no template_ids parameter)
- **Word Guess Detection**: Existing 20Q environment has sophisticated detection that:
  - Only detects wins on affirmative oracle responses ("Yes")
  - Uses word-boundary regex matching for exact whole word/phrase matches
  - Handles word variants, punctuation, and multi-word variants
  - Extracts questions from verbose agent responses

### Conversation Architecture
- **State Management**: Uses Conversation objects for parallel game tracking
- **LLM Oracle**: HuggingFace model (default: Llama-3.1-8B-Instruct) for answering yes/no questions
- **Parallel Processing**: Multiple independent games running simultaneously
- **Debug Logging**: Comprehensive debug output for tracing issues

### Template System
- **Existing Templates**: 20Q has templates like `twentyq_default_simple`, `twentyq_category_then_refine`
- **ATV Consideration**: ATV generates latent subgoals rather than explicit template IDs
- **Recommendation**: There is no clean way to recover templates that make any sense. For ATV, this feature is not applicable.

## Design Requirements

### 1. Core Architecture Requirements

#### 1.1 ATV 20 Questions Trainer Class
**File**: `src/atv_20q_agent_trainer.py`

```python
class ATV20QAgentTrainer(MCQAgentTrainer):
    """
    20 Questions Agent Trainer with ATV support.
    
    Extends MCQAgentTrainer to handle ATV-based high-level policies
    for the 20 Questions environment with question/guess detection.
    """
```

**Key Requirements**:
- Inherit from `MCQAgentTrainer` (already supports 20Q via `is_twenty_questions_env()`)
- Implement ATV-specific action detection for questions vs. word guesses
- Handle 20Q-specific environment integration (no template_ids in step)
- Maintain compatibility with existing 20Q reward structure

#### 1.2 Action Detection Patterns
Use the existing word guess detection from the 20Q environment, which:
- Detects wins only on affirmative oracle responses ("Yes")
- Uses word-boundary regex matching for exact whole word/phrase matches
- Handles word variants, punctuation, and multi-word variants
- Extracts questions from verbose agent responses

#### 1.3 Template Mapping for 20Q
Template mapping exists in the 20Q design (`twentyq_default_simple`, `twentyq_category_then_refine`, etc.) but may not be necessary for ATV since ATV generates latent subgoals rather than explicit template IDs. This should be an optional feature with an off-by-default switch.

### 2. Environment Integration Requirements

#### 2.1 Environment Detection and Handling
**Requirements**:
- Detect 20Q environment using existing `is_twenty_questions_env()` function
- Handle different reset methods (both support batch_size parameter)
- Handle different step methods (20Q doesn't use template_ids parameter)
- Maintain backward compatibility with MCQ environments

#### 2.2 State Management
**Requirements**:
- Handle 20Q state format: `[game_prompt, *conversation_turns]`
- Process conversation history for ATV generation
- Maintain conversation objects for parallel game tracking
- Support both question-answer pairs and word guesses

### 3. ATV Generation Requirements

#### 3.1 ATV Input Processing
**Requirements**:
- Process 20Q conversation state for ATV generation
- Extract game context (questions asked, answers received, current strategy)
- Handle variable-length conversation histories
- Maintain game metadata (question count, episode status)

#### 3.2 ATV Output Adaptation
**Requirements**:
- Generate ATV vectors appropriate for questioning strategies
- Support different ATV patterns for different game phases (early vs. late questions)
- Handle ATV conditioning for both question generation and word guessing
- Maintain ATV injection compatibility with existing low-level policy

### 4. Training Requirements

#### 4.1 Rollout Collection
**Requirements**:
- Collect rollout data compatible with 20Q environment
- Handle different action types (questions vs. guesses)
- Process 20Q-specific rewards (-1 per question, 0 for wins)
- Maintain ATV-specific data collection (ATV vectors, action detection)

#### 4.2 Training Loop Integration
**Requirements**:
- Integrate with existing dual-reward training structure
- Support ATV high-level policy training
- Maintain compatibility with existing critic training
- Handle 20Q-specific reward processing

## Implementation Plan

### Phase 1: Core ATV 20Q Trainer (Week 1-2)

#### 1.1 ATV20QAgentTrainer Class
**File**: `src/atv_20q_agent_trainer.py`

```python
class ATV20QAgentTrainer(MCQAgentTrainer):
    """
    20 Questions Agent Trainer with ATV support.
    """
    
    # 20Q-specific patterns
    WORD_GUESS_PATTERN = r"(?:guess|think|believe|word is|it's|it is)\s+(?:that\s+)?(?:the\s+)?(?:word\s+is\s+)?([a-zA-Z]+)"
    QUESTION_PATTERN = r"^[^.!?]*\?[^.!?]*$"
    
    # 20Q template mapping
    SUBGOAL_TO_TEMPLATE_20Q = {
        "category_question": "twentyq_category_then_refine",
        "property_question": "twentyq_property_elimination",
        "size_question": "twentyq_default_simple",
        "final_guess": "twentyq_final_guess",
        # ... more mappings
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_twentyq = True  # Override for 20Q-specific behavior
        print("Initialized ATV20QAgentTrainer for 20 Questions environment")
    
    def _detect_word_guesses(self, responses: List[str]) -> List[bool]:
        """Detect word guess actions in 20Q responses."""
        
    def _detect_questions(self, responses: List[str]) -> List[bool]:
        """Detect question actions in 20Q responses."""
        
    def _extract_guessed_words(self, responses: List[str]) -> List[str]:
        """Extract guessed words from responses."""
        
    def _generate_template_ids_from_subgoals_20q(self, actions: List[str]) -> List[str]:
        """Generate 20Q-specific template IDs from subgoals."""
```

#### 1.2 Action Detection Methods
**Implementation Requirements**:
No need for submit action detection, because every LL action is submitted.

### Phase 2: Environment Integration (Week 2-3)

#### 2.1 Environment Handling
**Requirements**:
- Override `collect_rollout()` to handle 20Q-specific environment calls
- Handle reset methods (both environments support batch_size parameter)
- Handle different step methods (no template_ids for 20Q)
- Maintain environment detection and compatibility

#### 2.2 State Processing
**Requirements**:
- Process 20Q conversation state for ATV generation
- Handle variable-length conversation histories
- Extract game context and metadata
- Support both question-answer pairs and word guesses

### Phase 3: ATV Integration (Week 3-4)

#### 3.1 ATV Generation for 20Q
**Requirements**:
- Adapt ATV generation for 20Q conversation context
- Support different ATV patterns for different game phases
- Handle 20Q-specific subgoal generation
- Maintain ATV injection compatibility

#### 3.2 Training Integration
**Requirements**:
- Integrate ATV training with 20Q reward structure
- Support dual-reward training for 20Q
- Handle ATV-specific training data collection
- Maintain compatibility with existing training loops

### Phase 4: Testing and Validation (Week 4-5)

#### 4.1 Unit Testing
no testing or validation required. we will fix issues on an ad hoc basis.

## Configuration Requirements

### 1. ATV 20Q Configuration
**File**: `config/atv_20q_config.py`

```python
@dataclass
class ATV20QConfig:
    """Configuration for ATV 20 Questions training."""
    
    # ATV configuration
    atv_model_name: str = "gpt2"
    atv_hidden_dim: int = 768
    atv_expansion_dim: int = 4096
    
    # 20Q-specific configuration
    max_questions: int = 20
    word_list_path: str = "data/twenty_questions_words.txt"
    
    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 10
    
    # Action detection configuration
    word_guess_threshold: float = 0.8
    question_confidence_threshold: float = 0.7
```
