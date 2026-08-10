# MCQ Agent Trainer Refactoring Plan

## Overview

This document outlines a comprehensive refactoring plan to transform the monolithic `MCQAgentTrainer` and `SingleAgentArcherTrainer` classes into a modular, maintainable, and testable architecture.

## Current Problems

### 1. Monolithic Classes
- `MCQAgentTrainer`: ~4,700 lines of code
- `SingleAgentArcherTrainer`: ~1,300 lines of code
- Single file contains multiple responsibilities

### 2. Mixed Responsibilities
- Data collection and processing
- Training logic (PPO, critics, actors)
- Reward computation (Skywork integration)
- GPU memory management
- Environment-specific logic
- Batch processing and validation

### 3. Code Duplication
- Multiple similar training methods with slight variations
- Repeated chunked training patterns
- Duplicate batch processing logic

### 4. Complex Nested Functions
- Large methods with deeply nested helper functions
- Difficult to understand and maintain
- Hard to unit test individual components

### 5. Poor Separation of Concerns
- Environment-specific logic mixed with general training logic
- Reward computation scattered throughout training methods
- GPU management mixed with training logic

## Proposed Modular Architecture

### Core Principles
1. **Single Responsibility Principle**: Each class has one clear purpose
2. **Composition over Inheritance**: Use composition to build complex trainers
3. **Dependency Injection**: Inject dependencies for better testability
4. **Interface Segregation**: Small, focused interfaces
5. **Fail Fast and Loud**: Maintain research codebase principles

## New File Structure

```
src/
├── training/
│   ├── __init__.py
│   ├── base_trainer.py              # Abstract base trainer
│   ├── ppo_trainer.py               # PPO training logic
│   ├── chunked_trainer.py           # Memory-efficient chunked training
│   ├── training_coordinator.py      # Training orchestration
│   ├── critics/
│   │   ├── __init__.py
│   │   ├── dual_reward_trainer.py   # HL/LL dual reward training
│   │   ├── single_agent_trainer.py  # Single agent critic training
│   │   └── critic_utils.py          # Common critic utilities
│   └── actors/
│       ├── __init__.py
│       ├── actor_trainer.py         # Base actor training
│       ├── single_agent_actor.py    # Single agent actor training
│       └── hierarchical_actor.py    # Hierarchical actor training
├── data/
│   ├── __init__.py
│   ├── rollout_collector.py         # Rollout collection logic
│   ├── conversation_data.py         # Conversation data structures
│   ├── batch_processor.py           # Batch processing utilities
│   └── data_validator.py            # Data validation
├── rewards/
│   ├── __init__.py
│   ├── reward_computer.py           # Base reward computation
│   ├── skywork_rewards.py           # Skywork model integration
│   ├── marginal_rewards.py          # Marginal reward decomposition
│   └── reward_aggregator.py         # Reward aggregation
├── environments/
│   ├── __init__.py
│   ├── mcq_environment.py           # MCQ environment integration
│   ├── twenty_questions_env.py      # 20Q specific logic
│   └── environment_factory.py       # Environment creation
├── trainers/
│   ├── __init__.py
│   ├── mcq_trainer.py               # Main MCQ trainer (orchestrator)
│   ├── single_agent_archer.py       # Single agent archer trainer
│   └── hierarchical_trainer.py      # Hierarchical training coordinator
└── utils/
    ├── __init__.py
    ├── gpu_manager.py                # GPU memory management
    └── training_utils.py             # Common training utilities
```

## Component Responsibilities

### 1. Training Components (`src/training/`)

#### `base_trainer.py`
- Abstract base class for all trainers
- Common training utilities and interfaces
- Standard training step orchestration

#### `ppo_trainer.py`
- PPO-specific training logic
- Advantage computation
- Policy gradient calculations

#### `chunked_trainer.py`
- Memory-efficient chunked training utilities
- Batch splitting and processing
- GPU memory management integration

#### `training_coordinator.py`
- Orchestrates training steps
- Manages training loops
- Coordinates between different components

### 2. Critic Training (`src/training/critics/`)

#### `dual_reward_trainer.py`
- Handles dual reward streams (HL/LL)
- High-level and low-level critic training
- Reward signal separation

#### `single_agent_trainer.py`
- Single agent critic training
- Combined critic objectives
- Archer-specific training logic

#### `critic_utils.py`
- Common critic utilities
- Value function computations
- Critic loss calculations

### 3. Actor Training (`src/training/actors/`)

#### `actor_trainer.py`
- Base actor training logic
- Policy gradient computations
- Actor loss calculations

#### `single_agent_actor.py`
- Single agent actor training
- Combined actor objectives
- Archer-specific actor logic

#### `hierarchical_actor.py`
- Hierarchical actor training
- High-level policy training
- Multi-level policy coordination

### 4. Data Management (`src/data/`)

#### `rollout_collector.py`
- Rollout collection logic
- Environment interaction
- Episode management

#### `conversation_data.py`
- Conversation data structures
- Data serialization/deserialization
- Conversation state management

#### `batch_processor.py`
- Batch processing utilities
- Data padding and alignment
- Batch validation

#### `data_validator.py`
- Data validation and safety checks
- Input sanitization
- Error detection

### 5. Reward Systems (`src/rewards/`)

#### `reward_computer.py`
- Base reward computation interface
- Reward aggregation utilities
- Reward validation

#### `skywork_rewards.py`
- Skywork model integration
- External reward model loading
- Reward model inference

#### `marginal_rewards.py`
- Marginal reward decomposition
- Token-level reward computation
- Reward attribution

#### `reward_aggregator.py`
- Reward aggregation logic
- Multi-reward combination
- Reward normalization

### 6. Environment Integration (`src/environments/`)

#### `mcq_environment.py`
- MCQ environment integration
- Environment-specific logic
- MCQ-specific utilities

#### `twenty_questions_env.py`
- 20 questions specific logic
- 20Q environment handling
- 20Q-specific utilities

#### `environment_factory.py`
- Environment creation and configuration
- Environment type detection
- Environment setup utilities

### 7. Specialized Trainers (`src/trainers/`)

#### `mcq_trainer.py`
- Main MCQ trainer (orchestrator)
- Coordinates all training components
- Maintains backward compatibility

#### `single_agent_archer.py`
- Single agent archer trainer
- Archer-specific training logic
- Single agent coordination

#### `hierarchical_trainer.py`
- Hierarchical training coordinator
- Multi-level training orchestration
- Hierarchical policy management

## Implementation Plan

### Phase 1: Extract Core Components (Week 1-2)

#### 1.1 Extract Data Structures
- [ ] Move `ConversationData` class to `src/data/conversation_data.py`
- [ ] Create `RolloutBatch` and `TrainingBatch` data classes
- [ ] Extract batch processing utilities to `src/data/batch_processor.py`
- [ ] Create data validation utilities

#### 1.2 Extract Reward System
- [ ] Move Skywork integration to `src/rewards/skywork_rewards.py`
- [ ] Extract marginal reward computation to `src/rewards/marginal_rewards.py`
- [ ] Create reward aggregation utilities
- [ ] Implement reward validation

#### 1.3 Extract Training Utilities
- [ ] Move chunked training methods to `src/training/chunked_trainer.py`
- [ ] Extract GPU memory management to `src/utils/gpu_manager.py`
- [ ] Create training step utilities
- [ ] Implement training coordination

### Phase 2: Create Specialized Trainers (Week 2-3)

#### 2.1 Critic Trainers
- [ ] Extract dual reward critic training to `src/training/critics/dual_reward_trainer.py`
- [ ] Extract single agent critic training to `src/training/critics/single_agent_trainer.py`
- [ ] Create common critic utilities
- [ ] Implement critic training coordination

#### 2.2 Actor Trainers
- [ ] Extract actor training logic to `src/training/actors/`
- [ ] Separate hierarchical vs single agent actor training
- [ ] Create actor training utilities
- [ ] Implement actor training coordination

#### 2.3 Environment Integration
- [ ] Extract environment-specific logic to `src/environments/`
- [ ] Create environment factory pattern
- [ ] Separate 20Q and MCQ environment handling
- [ ] Implement environment utilities

### Phase 3: Create High-Level Coordinators (Week 3-4)

#### 3.1 Main Trainers
- [ ] Create new `MCQTrainer` that orchestrates components
- [ ] Create new `SingleAgentArcherTrainer` using composition
- [ ] Maintain backward compatibility with existing interfaces
- [ ] Implement trainer configuration

#### 3.2 Training Coordinator
- [ ] Create `TrainingCoordinator` to manage training steps
- [ ] Implement strategy pattern for different training modes
- [ ] Add comprehensive logging and monitoring
- [ ] Implement training state management

### Phase 4: Testing and Validation (Week 4-5)

#### 4.1 Unit Tests
- [ ] Test each extracted component independently
- [ ] Create integration tests for the new architecture
- [ ] Validate backward compatibility
- [ ] Implement performance benchmarks

#### 4.2 Performance Validation
- [ ] Ensure no performance regression
- [ ] Optimize memory usage with new chunked approach
- [ ] Validate training convergence
- [ ] Implement performance monitoring

## Migration Strategy

### Backward Compatibility
- Keep existing `MCQAgentTrainer` as a thin wrapper
- Maintain all existing public interfaces
- Use deprecation warnings for old methods
- Provide migration guide for users

### Gradual Migration
- Allow both old and new architectures to coexist
- Use configuration flags to choose between architectures
- Implement feature flags for new components
- Provide migration utilities

### Testing Strategy
- Comprehensive unit tests for each component
- Integration tests for the complete training pipeline
- Performance regression tests
- Backward compatibility tests

## Key Benefits

### 1. Maintainability
- Smaller, focused files that are easier to understand
- Clear separation of concerns
- Reduced code duplication
- Better error isolation

### 2. Testability
- Components can be unit tested independently
- Mock dependencies for isolated testing
- Integration tests for component interactions
- Performance benchmarks for optimization

### 3. Extensibility
- Easy to add new training strategies
- Simple to integrate new environments
- Straightforward to add new reward types
- Flexible configuration system

### 4. Performance
- Better memory management with chunked training
- Optimized GPU memory usage
- Reduced memory fragmentation
- Improved training efficiency

### 5. Debugging
- Easier to isolate and fix issues
- Better error messages and logging
- Component-level debugging
- Performance profiling per component

## Implementation Priority

### High Priority (Immediate Impact)
1. **Reward System Extraction**: Skywork integration is complex and self-contained
2. **Data Structure Extraction**: Used everywhere, high impact on maintainability
3. **GPU Memory Management**: Critical for performance and stability

### Medium Priority (Core Functionality)
1. **Critic Training Logic**: Core training functionality
2. **Actor Training Logic**: Core training functionality
3. **Training Coordinator**: Orchestration and coordination

### Lower Priority (Polish)
1. **Environment Integrations**: Environment-specific optimizations
2. **Comprehensive Logging**: Monitoring and debugging improvements
3. **Advanced Configuration**: Enhanced configurability

## Success Metrics

### Code Quality
- [ ] Reduce average file size from 3000+ lines to <500 lines
- [ ] Achieve >90% test coverage for new components
- [ ] Eliminate code duplication
- [ ] Improve code readability scores

### Performance
- [ ] Maintain or improve training speed
- [ ] Reduce memory usage by 20%
- [ ] Improve GPU memory efficiency
- [ ] Maintain training convergence

### Maintainability
- [ ] Reduce time to add new training strategies
- [ ] Improve debugging efficiency
- [ ] Reduce onboarding time for new developers
- [ ] Increase code reusability

## Conclusion

This refactoring plan transforms the monolithic MCQ trainer classes into a clean, modular architecture that follows software engineering best practices while maintaining the research codebase principles of "fail fast and loud." The new architecture will be more maintainable, testable, and extensible while preserving all existing functionality and performance characteristics.

The phased implementation approach ensures minimal disruption to ongoing research while providing immediate benefits as components are extracted and improved. The backward compatibility strategy allows for a smooth transition without breaking existing workflows.
