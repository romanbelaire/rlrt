# Hierarchical 20 Questions Agent Architecture Plan

## Overview

This document outlines the plan to extend the 20 Questions environment to support a hierarchical agent architecture, similar to the existing MCQ (Hellaswag/ARC) implementation. The hierarchical approach will feature a **high-level policy** that performs meta-cognitive reasoning to guide a **low-level policy** that generates actual questions.

## Current 20 Questions Architecture

### Single-Level Agent (Current State)
- **Agent Role**: Directly generates yes/no questions to identify secret word
- **Environment**: Conversation-based with LLM oracle (Llama-3.1-8B-Instruct)
- **State Management**: Conversation objects track game history per parallel game
- **Rewards**: -1.0 per question, 0.0 for correct guess (encourages efficiency)
- **Templates**: Simple question generation strategies (`twentyq_default_simple`, `twentyq_category_then_refine`)

### Integration Points
- **Trainer**: `MCQAgentTrainer` already supports dual-agent training
- **Environment Detection**: `is_twenty_questions_env()` helper function
- **Template System**: Basic templates in `templates_general.json`
- **Conversation Architecture**: Uses `Conversation` objects for state management

## Proposed Hierarchical Architecture

### High-Level Policy (Meta-Cognitive Reasoning)
**Purpose**: Analyze game state and provide strategic guidance to low-level policy

**Core Responsibilities**:
1. **Information Assessment**: Evaluate what has been learned so far
2. **Strategy Selection**: Choose questioning approach (category, properties, elimination)
3. **Confidence Evaluation**: Determine if ready to make final guess
4. **Guidance Generation**: Provide specific instructions to low-level agent

**Example High-Level Actions**:
The high level action of the HL agent will be a free form text generation, not the template-based thinking like we have in MCQ. We need to write a robust system prompt for the HL agent, providing a structured summary of what the conversation has been so far, and a list of types of guidance it can give. The exmaple guidance will be similar to the templates, but much shorter, i.e. "Provide an instruction to your partner to do one of the following: Ask a question to gain more information; Make a final guess about the mystery word; ... etc"

### Low-Level Policy (Question Generation)
**Purpose**: Generate specific yes/no questions based on high-level guidance

**Core Responsibilities**:
1. **Question Formulation**: Craft clear, strategic yes/no questions
2. **Context Integration**: Use conversation history and high-level guidance
3. **Question Quality**: Ensure questions are answerable and informative
4. **Guess Execution**: Make final word guesses when instructed

**Conditioning Process**:
```
High-Level Instruction: "Focus on determining the broad category (living vs non-living)"
Conversation Context: [Previous Q&A pairs]
Generated Question: "Is it a living thing?"
```

## Implementation Plan

### Phase 1: write one single, robust system prompt for the HL agent. then, format the state for the LL agent that includes the conversation history as well as the HL instruction.

HL prompt: "You are one member a pair of agents playing 20 questions as a team. Your role is to provide guidance on how to think, and your partner uses that guidance to ask questions to the oracle. " (needs to be expanded upon) + "Context so far: {context} \n\n Provide an instruction to your partner to do one of the following: 
1) Ask a question to gain more information;
2) Make a final guess about the mystery word;" (also needs to be expanded upon)

LL Prompt: needs to be written



### Phase 2: Environment Integration

#### 2.1 Enhanced State Management
- **Current**: `[game_prompt, question1, answer1, question2, answer2, ...]`
- **Enhanced**: use the conversation object in @base.py to manage this. we should extend it with a child class.
The conversation class has attributes such as
        self.conversation_id = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        self.user_messages: List[str] = []
        self.assistant_messages: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.high_level_instructions: List[str] = []
        self.template_ids: List[str] = []  # Track template ID for each user message
        self.original_harmful_prompt: Optional[str] = None

We can add new attributes to track the word label, and other 20-questions specific information

```python
# Enhanced conversation metadata
conversation.set_metadata("questions_asked", 5)
conversation.set_metadata("categories_explored", ["living", "size", "location"])
conversation.set_metadata("confidence_level", "medium")
conversation.set_metadata("contradictions_detected", False)
```

#### 2.2 Dual Reward System
- **High-Level Rewards**: Correctness of final guess (-1 per turn unless a win is detected, then 0)
- **Low-Level Rewards**: use a judge similar to MCQ to assess if the LL agent is following instructions or not.

```python
def compute_hierarchical_rewards(self, conversation_state, action_type):
    """Compute separate rewards for HL and LL policies"""
    if action_type == "high_level":
        # Reward based on game score
        return self._compute_strategy_reward(conversation_state)
    else:
        # Reward based on instruction following 
        return self._compute_question_reward(conversation_state)
```

### Phase 3: Training Integration

#### 3.1 Trainer Modifications
- **Extend** `MCQAgentTrainer.collect_rollout()` for 20Q hierarchical support
- **Add** 20Q-specific high-level policy action selection
- **Implement** dual reward collection for 20Q tasks

```python
# In collect_rollout()
if is_twenty_questions and use_high_level_policy:
    # High-level policy selects strategy
    hl_actions, hl_log_probs = high_level_policy.act(states)
    
    # Low-level policy generates questions based on strategy
    ll_actions, ll_log_probs = low_level_policy.act(states, hl_actions)
    
    # Collect dual rewards
    hl_rewards = environment.compute_strategy_rewards(hl_actions, states)
    ll_rewards = environment.compute_question_rewards(ll_actions, states)
```

#### 3.2 High-Level Policy Implementation
- **free text generation**: LLM-based instruction generation


## Clarifying Questions & Design Decisions

### 1. High-Level Policy Architecture
**Question**: Should we use template-based selection (like MCQ) or LLM-based instruction generation?


- **LLM-based**: Generate custom instructions based on game state
use one single system prompt



### 2. Reward Structure Design
**Question**: How should we structure rewards to train both policies effectively?

use a reward structure similar to the MCQ environment. Task rewards are used for HL, instruction-following score used for LL.

### 3. State Representation
**Question**: How should we represent game state for the high-level policy?

use the conversation data structure class, as discussed above.

### 4. Integration Complexity
**Question**: Should we modify existing 20Q environment or create new hierarchical version?

create a new environment class for modularity. reuse agent code when possible, but don't modify old classes. if needed, create child classes.

### 5. Training Modes
**Question**: Should we support different training modes like MCQ?

**Proposed Modes**:
1. **Dual Training**: Train both HL and LL policies simultaneously
2. **HL-Only**: Train only high-level policy (freeze LL)
3. **LL-Only**: Train only low-level policy (fixed HL strategies)

yes

**Configuration Example**:
```json
{
  "training": {
    "hierarchical_20q_mode": "default",  // "default", "hl_only", "ll_only",
    "freeze_ll_policy": false,
    "use_high_level_policy": true,
    "high_level_policy_type": "template_dqn"
  }
}
```

---

*This plan provides a comprehensive roadmap for implementing hierarchical 20 Questions agents. The modular approach allows for incremental development and validation while maintaining compatibility with existing systems.*
