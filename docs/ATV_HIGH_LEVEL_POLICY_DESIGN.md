# Adaptive Task Vector (ATV) High-Level Policy Design Document

## Overview

This document outlines the design and implementation of an Adaptive Task Vector (ATV) based high-level policy that provides subgoals to the low-level policy via latent representations. The ATV approach enables dynamic generation of task-specific vectors conditioned on each input query, allowing the high-level policy to adapt its behavior based on the specific context and requirements of each conversation turn.

## Background

### Current Hierarchical RL Architecture

The existing codebase implements a hierarchical reinforcement learning system with:

1. **High-Level Policy**: Selects strategic templates/instructions (DQN, PPO, or LLM-based)
2. **Low-Level Policy**: Generates responses conditioned on high-level instructions
3. **Critics**: Evaluate value functions for both levels
4. **Environment**: Simulates conversations and provides rewards

### ATV Paper Insights

The ATV paper introduces a framework that:
- Dynamically generates task vectors conditioned on each input query
- Uses a small language model to generate intermediate task representations
- Transforms these representations to match the target LLM's architecture
- Injects task vectors into the LLM's hidden states for steering

## Design Goals

1. **Dynamic Subgoal Generation**: Generate context-aware subgoals rather than fixed templates
2. **Latent Representation Learning**: Learn meaningful latent representations for task guidance
3. **Input-Conditioned Adaptation**: Adapt behavior based on specific input characteristics
4. **Efficient Integration**: Seamlessly integrate with existing hierarchical RL framework
5. **Interpretability**: Maintain interpretability of high-level decisions

## Architecture Design

### 1. ATV High-Level Policy Components

```
ATVHighLevelPolicy
├── Small Language Model (M_small)
│   ├── Input: Conversation state + context
│   ├── Output: Compact task representation (v_small)
│   └── Architecture: GPT-2 variant or similar
├── ATV Expansion Module (f_θ)
│   ├── Input: v_small (R^d_s)
│   ├── Output: ATV (R^L×d_l)
│   └── Architecture: Linear projection + reshaping
└── ATV Injection Module
    ├── Input: ATV + Low-level policy hidden states
    ├── Output: Modified hidden states
    └── Method: Additive injection with scaling
```

### 2. Integration with Existing Framework

```
Current Flow:
State → High-Level Policy → Instructions → Low-Level Policy → Response

ATV Flow:
State → ATV High-Level Policy → Latent Subgoals → Low-Level Policy → Response
                ↓
        [ATV Generation & Injection]
```

### 3. Key Design Decisions

#### 3.1 ATV Generation Strategy
- **Small Model**: Use GPT-2 or similar lightweight model for task representation generation
- **Input Encoding**: Encode conversation state + original question + context
- **Output**: Compact vector representation (v_small ∈ R^d_s)

#### 3.2 ATV Expansion Architecture
- **Linear Projection**: Single linear layer from R^d_s to R^(L×d_l)
- **Reshaping**: Reshape to match target model's layer structure
- **No Activation**: Keep expansion linear for interpretability

#### 3.3 ATV Injection Method
- **Additive Injection**: h̃_l = h_l + λv_l_ATV
- **Layer-wise**: Inject different ATV slices into different layers
- **Scaling**: Use learnable or fixed scaling factor λ

## Implementation Plan

### Phase 1: Core ATV Components

#### 1.1 ATV High-Level Policy Class
**File**: `src/atv_high_level_policy.py`

```python
class ATVHighLevelPolicy(nn.Module):
    """
    Adaptive Task Vector based high-level policy.
    
    Generates dynamic task vectors conditioned on input queries
    to provide latent subgoals to the low-level policy.
    """
    
    def __init__(
        self,
        small_model_config: Dict[str, Any],
        target_model_config: Dict[str, Any],
        expansion_config: Dict[str, Any],
        injection_config: Dict[str, Any]
    ):
        # Initialize small language model
        # Initialize expansion module
        # Initialize injection parameters
        
    def generate_atv(self, state_texts: List[str], base_questions: Optional[List[str]] = None) -> torch.Tensor:
        """Generate ATV vectors for given states."""
        
    def inject_atv(self, low_level_policy: nn.Module, atv_vectors: torch.Tensor) -> None:
        """Inject ATV vectors into low-level policy."""
        
    def act(self, state_texts: List[str], base_questions: Optional[List[str]] = None) -> Tuple[List[str], torch.Tensor]:
        """Generate latent subgoals and return ATV vectors."""
```

#### 1.2 ATV Expansion Module
**File**: `src/atv_expansion.py`

```python
class ATVExpansionModule(nn.Module):
    """
    Expands compact task representations to full ATV vectors.
    """
    
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int):
        # Linear projection layer
        # Reshaping logic
        
    def forward(self, compact_vectors: torch.Tensor) -> torch.Tensor:
        """Expand compact vectors to full ATV format."""
```

#### 1.3 ATV Injection Module
**File**: `src/atv_injection.py`

```python
class ATVInjectionModule:
    """
    Handles injection of ATV vectors into target model.
    """
    
    def __init__(self, injection_layers: List[int], scaling_factor: float):
        # Layer selection
        # Scaling configuration
        
    def inject(self, model: nn.Module, atv_vectors: torch.Tensor) -> None:
        """Inject ATV vectors into model hidden states."""
        
    def remove_injection(self, model: nn.Module) -> None:
        """Remove ATV injection hooks."""
```

### Phase 2: Training Infrastructure

#### 2.1 ATV Training Loop
**File**: `src/atv_trainer.py`

```python
class ATVTrainer:
    """
    Training infrastructure for ATV high-level policy.
    """
    
    def train_atv_policy(
        self,
        atv_policy: ATVHighLevelPolicy,
        low_level_policy: nn.Module,
        training_data: List[Dict],
        num_epochs: int
    ) -> None:
        """Train ATV policy with supervised learning."""
        
    def compute_atv_loss(
        self,
        atv_vectors: torch.Tensor,
        target_outputs: List[str],
        low_level_policy: nn.Module
    ) -> torch.Tensor:
        """Compute loss for ATV training."""
```

#### 2.2 ATV Data Collection
**File**: `src/atv_data_collection.py`

```python
class ATVDataCollector:
    """
    Collects training data for ATV policy.
    """
    
    def collect_atv_training_data(
        self,
        conversations: List[SemanticConversation],
        target_model: nn.Module
    ) -> List[Dict]:
        """Collect training data from conversation history."""
```

### Phase 3: Integration with Existing Framework

#### 3.1 Modified Low-Level Policy
**File**: `src/atv_conditioned_policy.py`

```python
class ATVConditionedLowLevelPolicy(LowLevelPolicy):
    """
    Low-level policy that accepts ATV vectors for conditioning.
    """
    
    def __init__(self, base_policy: LowLevelPolicy, atv_injection: ATVInjectionModule):
        # Initialize with base policy
        # Add ATV injection capability
        
    def act_with_atv(
        self,
        state: List[str],
        atv_vectors: torch.Tensor,
        **kwargs
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Generate responses conditioned on ATV vectors."""
```

#### 3.2 Updated Trainer Integration
**File**: `src/mcq_agent_trainer.py` (modifications)

```python
class MCQAgentTrainer(AgentTrainer):
    """
    Extended trainer with ATV support.
    """
    
    def train_atv_hierarchical_policy(
        self,
        atv_high_level_policy: ATVHighLevelPolicy,
        atv_conditioned_low_level_policy: ATVConditionedLowLevelPolicy,
        training_data: List[Dict]
    ) -> None:
        """Train hierarchical policy with ATV."""
```

### Phase 4: Configuration and Utilities

#### 4.1 ATV Configuration
**File**: `config/atv_config.py`

```python
@dataclass
class ATVConfig:
    """Configuration for ATV high-level policy."""
    
    # Small model configuration
    small_model_name: str = "gpt2"
    small_model_hidden_dim: int = 768
    
    # Expansion configuration
    expansion_input_dim: int = 768
    expansion_output_dim: int = 4096  # L × d_l
    
    # Injection configuration
    injection_layers: List[int] = field(default_factory=lambda: [8, 16, 24, 32])
    scaling_factor: float = 0.1
    
    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 10
```

#### 4.2 ATV Utilities
**File**: `src/atv_utils.py`

```python
def create_atv_high_level_policy(config: ATVConfig) -> ATVHighLevelPolicy:
    """Factory function for creating ATV high-level policy."""
    
def load_atv_policy(path: str) -> ATVHighLevelPolicy:
    """Load trained ATV policy from disk."""
    
def save_atv_policy(policy: ATVHighLevelPolicy, path: str) -> None:
    """Save trained ATV policy to disk."""
```

## Training Strategy

### 1. Supervised Learning Phase

1. **Data Collection**: Collect conversation data with successful high-level instructions
2. **ATV Generation**: Train small model to generate task representations
3. **Expansion Training**: Train expansion module to map representations to ATV vectors
4. **End-to-End Training**: Joint training of both components

### 2. Reinforcement Learning Phase

1. **ATV Policy Training**: Train ATV policy using PPO with hierarchical rewards
2. **Low-Level Adaptation**: Fine-tune low-level policy to work with ATV conditioning
3. **Joint Optimization**: Optimize both policies together

### 3. Training Objectives

- **Supervised Loss**: Cross-entropy loss on generated subgoals
- **Reinforcement Loss**: PPO loss with hierarchical advantages
- **Regularization**: L2 regularization on ATV vectors
- **Consistency Loss**: Ensure ATV vectors are consistent across similar contexts

## Evaluation Metrics

### 1. Task Performance
- **Success Rate**: Percentage of successful task completions
- **Efficiency**: Number of turns to complete tasks
- **Quality**: LlamaGuard scores and human evaluation

### 2. ATV Quality
- **Diversity**: Diversity of generated ATV vectors
- **Consistency**: Consistency of ATV vectors for similar contexts
- **Interpretability**: Ability to interpret ATV vector meanings

### 3. Comparison Baselines
- **Template-based**: Current template selection methods
- **LLM-based**: Current LLM instruction generation
- **Hybrid**: Current hybrid approach

## Implementation Timeline

### Week 1-2: Core Components
- Implement ATVHighLevelPolicy class
- Implement ATVExpansionModule
- Implement ATVInjectionModule
- Basic unit tests

### Week 3-4: Training Infrastructure
- Implement ATVTrainer
- Implement ATVDataCollector
- Training loop integration
- Validation and testing

### Week 5-6: Framework Integration
- Modify existing low-level policy
- Update trainer classes
- Configuration system
- Integration testing

### Week 7-8: Evaluation and Optimization
- Performance evaluation
- Hyperparameter tuning
- Documentation
- Final testing

## Risk Mitigation

### 1. Technical Risks
- **ATV Injection Complexity**: Start with simple additive injection
- **Training Instability**: Use gradient clipping and careful learning rates
- **Memory Usage**: Implement efficient ATV storage and computation

### 2. Integration Risks
- **Backward Compatibility**: Maintain existing interfaces
- **Performance Degradation**: Implement fallback mechanisms
- **Training Complexity**: Start with supervised learning before RL

### 3. Evaluation Risks
- **Baseline Comparison**: Ensure fair comparison with existing methods
- **Metric Selection**: Use multiple evaluation metrics
- **Statistical Significance**: Use proper statistical testing

## Future Extensions

### 1. Advanced ATV Architectures
- **Multi-scale ATV**: Different ATV vectors for different time scales
- **Hierarchical ATV**: Nested ATV structures for complex tasks
- **Dynamic ATV**: ATV vectors that evolve during conversation

### 2. Enhanced Training Methods
- **Meta-Learning**: Learn to generate ATV vectors for new tasks
- **Adversarial Training**: Improve robustness of ATV generation
- **Multi-task Learning**: Train on multiple task types simultaneously

### 3. Interpretability and Analysis
- **ATV Visualization**: Visualize ATV vector meanings
- **Causal Analysis**: Understand causal relationships in ATV generation
- **Ablation Studies**: Systematic analysis of ATV components

## Conclusion

The ATV-based high-level policy design provides a novel approach to hierarchical reinforcement learning that addresses the limitations of fixed template-based methods. By dynamically generating task vectors conditioned on input queries, the system can adapt its behavior to specific contexts and learn meaningful latent representations for task guidance.

The implementation plan provides a structured approach to building this system while maintaining compatibility with the existing framework. The phased approach allows for incremental development and testing, reducing implementation risks while ensuring robust integration.

This design document serves as a comprehensive guide for implementing the ATV high-level policy and should be updated as the implementation progresses and new insights are gained.
