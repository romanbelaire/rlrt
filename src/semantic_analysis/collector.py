"""
Main data collection orchestrator for semantic analysis.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from types import SimpleNamespace
from transformers import AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

from .data_structures import ConversationTurn, SemanticConversation
from .vector_store import SemanticVectorStore
from src.config import AgentConfig
from src.environment import MultiStepEnv
from src.base import BaseModelSingleton
from src.critics import SequenceLevelValueCritic, TokenLevelValueCritic
# Removed unused imports - policies are loaded from checkpoints instead


class SemanticDataCollector:
    """Main data collection orchestrator for semantic analysis."""
    
    def __init__(self, config: AgentConfig, checkpoint_dir: Optional[str] = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.environment = None
        self.high_level_policy = None
        self.low_level_policy = None
        # Add critics and tokenizer handles for scoring
        self.sequence_critic = None
        self.token_critic = None
        self.tokenizer = None
        self.vector_store = None
        self.conversation_id_counter = 0
        self.singletons = {}
        
        # Initialize components
        self._setup_model_singletons()
        self._initialize_environment()
        self._initialize_policies()
        self._initialize_vector_store()
        
        # Create output directories
        self._create_output_dirs()
    
    def _setup_model_singletons(self):
        """Setup model singletons using the same logic as main.py and test_runner.py."""
        print("Setting up model singletons...")
        
        # MIRROR main.py/setup_model_singletons for device placement
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        agent_prefer_gpu = 0 if num_gpus > 0 else None
        target_prefer_gpu = 1 if num_gpus > 1 else 0 if num_gpus == 1 else None
        guard_prefer_gpu = 1 if num_gpus > 1 else 0 if num_gpus == 1 else None
        
        print(f"Number of available GPUs: {num_gpus}")
        print(f"Agent prefer GPU: {agent_prefer_gpu}")
        print(f"Target prefer GPU: {target_prefer_gpu}")
        print(f"Guard prefer GPU: {guard_prefer_gpu}")
        
        # Agent singleton (for policy)
        agent_singleton = BaseModelSingleton.get_instance(
            model_name=self.config.model.hf_checkpoint,
            use_offloading=self.config.training.use_offloading,
            config=self.config,
            prefer_gpu=agent_prefer_gpu,
            quantize=getattr(self.config.model, 'quantize', False)
        )
        
        # Tokenizer
        print(f"Loading tokenizer from: {self.config.model.hf_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.hf_checkpoint,
            use_fast=True,
            padding_side='left'
        )
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
        agent_singleton.tokenizer = tokenizer
        # Cache tokenizer on collector
        self.tokenizer = tokenizer
        
        agent_singleton.base_model.config.attn_implementation = "flash_attention_2"
        print(f"Using attention implementation flash attention 2 for {agent_singleton.model_name}")
        
        agent_singleton.base_model.eval()
        for name, param in agent_singleton.base_model.named_parameters():
            if "lora_" in name:
                continue
            param.requires_grad_(False)
        print("Agent backbone set to eval(); LoRA adapters remain trainable when present")
        
        # Target singleton
        target_singleton = None
        target_tokenizer = None
        if 'target_model' in self.config.environment.__dict__ and self.config.environment.target_model:
            target_model_name = self.config.environment.target_model
            openai_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-5"]
            is_openai_target = any(target_model_name.startswith(model) for model in openai_models)
            if not is_openai_target:
                # Load target directly onto GPU 1 (no device_map)
                target_tokenizer = AutoTokenizer.from_pretrained(
                    target_model_name,
                    use_fast=True,
                    padding_side='left'
                )
                target_tokenizer.padding_side = 'left'
                if target_tokenizer.pad_token is None:
                    if target_tokenizer.eos_token is not None:
                        target_tokenizer.pad_token = target_tokenizer.eos_token
                        target_tokenizer.pad_token_id = target_tokenizer.eos_token_id
                    else:
                        target_tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
                model_kwargs = {
                    'torch_dtype': torch.bfloat16,
                    'low_cpu_mem_usage': True,
                    'trust_remote_code': True,
                }
                target_model = AutoModelForCausalLM.from_pretrained(target_model_name, **model_kwargs)
                target_model = target_model.to('cuda:1' if torch.cuda.device_count() > 1 else 'cuda')
                if hasattr(target_model, 'config') and hasattr(target_model.config, 'attn_implementation'):
                    target_model.config.attn_implementation = 'flash_attention_2'
                target_singleton = SimpleNamespace(base_model=target_model, tokenizer=target_tokenizer)
                # Eval + freeze target model
                target_singleton.base_model.eval()
                for _p in target_singleton.base_model.parameters():
                    _p.requires_grad_(False)
         
        # Guard singleton
        guard_singleton = None
        guard_tokenizer = None
        if 'guard_model' in self.config.environment.__dict__ and self.config.environment.guard_model:
            guard_model_name = self.config.environment.guard_model
            guard_tokenizer = AutoTokenizer.from_pretrained(
                guard_model_name,
                use_fast=True,
                padding_side='left'
            )
            guard_tokenizer.padding_side = 'left'
            if guard_tokenizer.pad_token is None:
                if guard_tokenizer.eos_token is not None:
                    guard_tokenizer.pad_token = guard_tokenizer.eos_token
                    guard_tokenizer.pad_token_id = guard_tokenizer.eos_token_id
                else:
                    guard_tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
            guard_model = AutoModelForCausalLM.from_pretrained(
                guard_model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            guard_model = guard_model.to('cuda:1' if torch.cuda.device_count() > 1 else 'cuda')
            if hasattr(guard_model, 'config') and hasattr(guard_model.config, 'attn_implementation'):
                guard_model.config.attn_implementation = 'flash_attention_2'
            guard_singleton = SimpleNamespace(base_model=guard_model, tokenizer=guard_tokenizer)
            # Eval + freeze guard model
            guard_singleton.base_model.eval()
            for _p in guard_singleton.base_model.parameters():
                _p.requires_grad_(False)
        
        # Compose model_singletons dict as in main.py
        self.singletons = {
            "agent": agent_singleton,
            "tokenizer": tokenizer,
            "target": target_singleton,
            "target_tokenizer": target_tokenizer,
            "guard": guard_singleton,
            "guard_tokenizer": guard_tokenizer
        }
        
        # Debug singletons
        print(f"Singletons created:")
        print(f"  agent: {self.singletons['agent'] is not None}")
        print(f"  tokenizer: {self.singletons['tokenizer'] is not None}")
        print(f"  shared backbone: {agent_singleton.shared_model is not None}")
        
        # Check GPU memory after loading models
        if torch.cuda.is_available():
            print(f"GPU memory after model loading:")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}")
                print(f"    Total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
                print(f"    Allocated: {torch.cuda.memory_allocated(i) / 1e9:.1f} GB")
                print(f"    Cached: {torch.cuda.memory_reserved(i) / 1e9:.1f} GB")
            
            # Check which GPU each model is on
            print(f"Model device placement:")
            for model_name in ["agent", "target", "guard"]:
                model = self.singletons.get(model_name)
                if model is not None:
                    try:
                        model_device = next(model.base_model.parameters()).device
                        print(f"  {model_name.capitalize()} model: {model_device}")
                    except Exception as e:
                        print(f"  WARNING: Could not check device for {model_name} model: {e}")
    
    def _initialize_environment(self):
        """Initialize the environment for conversation processing."""
        print("Initializing environment...")
        
        # Get the HF token from the environment if available
        hf_token = os.environ.get("HF_TOKEN") or None
        
        # Create environment with model singletons
        self.environment = MultiStepEnv(
            config=self.config.environment,
            hf_token=hf_token,
            target_singleton=self.singletons.get("target"),
            target_tokenizer=self.singletons.get("target_tokenizer"),
            guard_singleton=self.singletons["guard"],
            guard_tokenizer=self.singletons["guard_tokenizer"],
            debug_mode=self.config.training.debug_mode,
            max_new_tokens=self.config.environment.max_new_tokens,
            target_quantize=getattr(self.config.environment, 'target_quantize', None),
            guard_quantize=getattr(self.config.environment, 'guard_quantize', None),
            evaluation_mode=False  # Use training data for semantic analysis
        )
        print(f"Environment initialized with max_steps={self.config.environment.max_steps}")
    
    def _initialize_policies(self):
        """Initialize hierarchical policies for scoring."""
        print("Initializing policies...")
        
        if self.checkpoint_dir:
            # Load policies from checkpoint
            print(f"Loading policies from checkpoint: {self.checkpoint_dir}")
            self._load_policies_from_checkpoint()
        else:
            # Semantic analysis requires an existing trained policy
            raise ValueError("Semantic analysis requires a checkpoint directory with trained policies. Please provide --checkpoint_dir argument.")

    def _load_policies_from_checkpoint(self):
        """Load policies from checkpoint directory."""
        from src.test_runner import _load_policy, _load_high_level_policy
        from src.critics import SequenceLevelValueCritic, TokenLevelValueCritic
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load low-level policy
        policy_dir = os.path.join(self.checkpoint_dir, "policy")
        if not os.path.exists(policy_dir):
            raise FileNotFoundError(f"Policy directory not found: {policy_dir}. Semantic analysis requires an existing trained policy.")
        
        print(f"Loading low-level policy from: {policy_dir}")
        self.low_level_policy = _load_policy(policy_dir, device=device)
        print("Low-level policy loaded successfully")
        
        # Load critics for scoring
        seq_dir = os.path.join(self.checkpoint_dir, "sequence_critic")
        tok_dir = os.path.join(self.checkpoint_dir, "token_critic")
        if not os.path.exists(seq_dir) or not os.path.exists(tok_dir):
            raise FileNotFoundError("Critic directories not found in checkpoint. Both sequence_critic and token_critic are required for semantic analysis scoring.")
        # Fail-fast verification of expected value head files
        seq_head_path = os.path.join(seq_dir, "sequence_value_head.pt")
        tok_head_path = os.path.join(tok_dir, "token_value_head.pt")
        if not os.path.exists(seq_head_path):
            raise FileNotFoundError(f"Missing sequence critic value head: {seq_head_path}")
        if not os.path.exists(tok_head_path):
            raise FileNotFoundError(f"Missing token critic value head: {tok_head_path}")
        
        shared_backbone = self.singletons["agent"].shared_model
        print(f"Loading sequence critic from: {seq_dir}")
        self.sequence_critic = SequenceLevelValueCritic(
            hf_checkpoint=self.config.model.hf_checkpoint,
            tokenizer=self.tokenizer,
            model=shared_backbone,
        )
        self.sequence_critic.load(seq_dir)
        print("Sequence critic loaded successfully")
        
        print(f"Loading token critic from: {tok_dir}")
        self.token_critic = TokenLevelValueCritic(
            hf_checkpoint=self.config.model.hf_checkpoint,
            tokenizer=self.tokenizer,
            model=shared_backbone,
        )
        self.token_critic.load(tok_dir)
        print("Token critic loaded successfully")
        
        # Load high-level policy if it exists
        high_level_policy_dir = os.path.join(self.checkpoint_dir, "high_level_policy")
        if os.path.exists(high_level_policy_dir):
            print(f"Loading high-level policy from: {high_level_policy_dir}")
            encoder = shared_backbone
            tokenizer = self.singletons["tokenizer"]
            if encoder is None:
                raise ValueError("Encoder from critic singleton is None - cannot load high-level policy")
            if tokenizer is None:
                raise ValueError("Tokenizer from singletons is None - cannot load high-level policy")
            self.high_level_policy = _load_high_level_policy(high_level_policy_dir, device=device, encoder=encoder, tokenizer=tokenizer)
            if self.high_level_policy is not None:
                print("High-level policy loaded successfully")
            else:
                raise RuntimeError("High-level policy loading returned None unexpectedly")
        else:
            if hasattr(self.config.model, 'use_high_level_policy') and self.config.model.use_high_level_policy:
                raise FileNotFoundError(f"High-level policy directory not found: {high_level_policy_dir}. Expected based on config.use_high_level_policy=True")
            else:
                print(f"High-level policy directory not found: {high_level_policy_dir}")
                print("Continuing without high-level policy (using low-level policy only)")
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store."""
        print("Initializing vector store...")
        
        # Get embedding dimension from target model
        if self.singletons.get("target") is not None:
            embedding_dim = self.singletons["target"].base_model.config.hidden_size
        else:
            embedding_dim = self.config.semantic_analysis.embedding_dim
        
        self.vector_store = SemanticVectorStore(
            embedding_dim=embedding_dim,
            index_type=self.config.semantic_analysis.faiss_index_type
        )
        print(f"Vector store initialized with embedding_dim={embedding_dim}")
    
    def _create_output_dirs(self):
        """Create output directories for data storage."""
        output_dir = self.config.semantic_analysis.output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "vectors"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "conversations"), exist_ok=True)
        print(f"Output directories created: {output_dir}")
    
    def collect_dataset(self, harmbench_data: pd.DataFrame, max_conversations: Optional[int] = None) -> None:
        """Main collection method for HarmBench dataset."""
        print(f"Starting data collection for {len(harmbench_data)} HarmBench entries")
        
        # Calculate how many conversations to generate per HarmBench question
        total_harmbench_questions = len(harmbench_data)
        target_conversations = max_conversations or total_harmbench_questions
        conversations_per_question = max(1, target_conversations // total_harmbench_questions)
        
        # If we have more capacity, distribute remaining conversations
        remaining_conversations = target_conversations - (conversations_per_question * total_harmbench_questions)
        
        print(f"Target: {target_conversations} total conversations")
        print(f"Will generate {conversations_per_question} conversations per HarmBench question")
        if remaining_conversations > 0:
            print(f"Plus {remaining_conversations} additional conversations distributed across questions")
        
        # Choose processing method based on batch size
        use_batched_processing = self.config.semantic_analysis.batch_size > 1
        print(f"Using {'batched' if use_batched_processing else 'sequential'} processing (batch_size={self.config.semantic_analysis.batch_size})")
        
        conversations = []
        metadata = {
            'total_conversations': 0,
            'successful_conversations': 0,
            'failed_conversations': 0,
            'category_stats': {},
            'statistics': {},
            'conversations_per_question': conversations_per_question,
            'total_harmbench_questions': total_harmbench_questions
        }
        
        # Process conversations in batches
        batch_size = self.config.semantic_analysis.batch_size
        save_interval = self.config.semantic_analysis.save_interval
        
        total_conversations_generated = 0
        
        # Create batches of questions and their variants
        conversation_queue = []
        for i, (_, row) in enumerate(harmbench_data.iterrows()):
            question = row[self.config.semantic_analysis.question_column]
            category = row.get(self.config.semantic_analysis.category_column, "unknown")
            
            # Determine how many conversations to generate for this question
            num_conversations_this_question = conversations_per_question
            if remaining_conversations > 0:
                num_conversations_this_question += 1
                remaining_conversations -= 1
            
            # Add conversation variants to queue
            for conv_variant in range(num_conversations_this_question):
                if total_conversations_generated >= target_conversations:
                    break
                
                conversation_queue.append({
                    'question': question,
                    'category': category,
                    'variant_id': conv_variant,
                    'harmbench_idx': i
                })
                total_conversations_generated += 1
                
            if total_conversations_generated >= target_conversations:
                break
        
        print(f"Created conversation queue with {len(conversation_queue)} conversations")
        
        # Process conversations in batches
        for batch_start in tqdm(range(0, len(conversation_queue), batch_size), desc="Processing conversation batches"):
            batch_end = min(batch_start + batch_size, len(conversation_queue))
            batch_items = conversation_queue[batch_start:batch_end]
            
            # Extract batch data
            batch_questions = [item['question'] for item in batch_items]
            batch_categories = [item['category'] for item in batch_items]
            batch_variant_ids = [item['variant_id'] for item in batch_items]
            
            # Process this batch of conversations concurrently
            batch_conversations = self._process_conversation_batch(
                batch_questions, batch_categories, batch_variant_ids
            )
            
            # Add successful conversations to results
            for i, conversation in enumerate(batch_conversations):
                conversations.append(conversation)
                metadata['successful_conversations'] += 1
                
                # Update category statistics
                category = batch_categories[i]
                if category not in metadata['category_stats']:
                    metadata['category_stats'][category] = 0
                metadata['category_stats'][category] += 1


            
            # Save intermediate results
            if len(conversations) > 0 and len(conversations) % save_interval == 0:
                self._save_intermediate_results(conversations, metadata)
        
        metadata['total_conversations'] = len(conversations) + metadata['failed_conversations']
        
        # Save final results
        self._save_final_results(conversations, metadata)
        print(f"Data collection completed. Generated {len(conversations)} successful conversations from {len(conversation_queue)} total attempts ({len(harmbench_data)} HarmBench questions)")
    
    def _process_single_conversation_with_temperature(self, question: str, category: str, variant_id: int = 0) -> Optional[SemanticConversation]:
        """Process one conversation with temperature variation for diversity.
        
        This method now uses the batched processing infrastructure for consistency.
        """
        # Use the batch method with a single conversation
        batch_results = self._process_conversation_batch([question], [category], [variant_id])
        return batch_results[0] if batch_results else None
    
    def _process_single_conversation(self, question: str, category: str) -> Optional[SemanticConversation]:
        """Process one conversation through N=30 turns."""
        conversation_id = f"conv_{self.conversation_id_counter:06d}"
        self.conversation_id_counter += 1
         
        # Set environment state for new conversation
        # The environment expects a list of conversation histories
        self.environment.set_state([[question]])
        
        turns = []
        max_turns = self.config.semantic_analysis.max_conversation_length
        
        for turn_num in range(max_turns):
            turn = self._process_single_turn(turn_num, question)
            turns.append(turn)
        
        if not turns:
            return None
        
        # Create conversation object
        conversation = SemanticConversation(
            conversation_id=conversation_id,
            original_question=question,
            category=category,
            turns=turns
        )
        
        # Add to vector store
        embedding_arrays = conversation.get_embedding_arrays()
        self.vector_store.add_conversation_vectors(
            conversation_id=conversation_id,
            action_embeddings=embedding_arrays['action_embeddings'],
            response_embeddings=embedding_arrays['response_embeddings'],
            context_embeddings=embedding_arrays['context_embeddings']
        )
        
        return conversation
    
    def _process_single_turn(self, turn_num: int, original_question: str) -> ConversationTurn:
        """Process a single conversation turn."""
        
        # Get current state and generate action
        conversation_histories = self.environment.get_agent_conversation_history()
        # Extract text content from conversation histories
        current_state = []
        for history in conversation_histories:
            # Extract content from each message in the history
            messages = [msg['content'] for msg in history]
            current_state.extend(messages)
        
        # Generate high-level instruction if using hierarchical policy
        if self.high_level_policy is None:
            raise RuntimeError("High-level policy must be present for semantic analysis. None was loaded.")
        high_level_instruction = self.high_level_policy.select_action(current_state)
        # Generate low-level action
        action = self._generate_action(current_state, high_level_instruction)
        
        # Get target response
        target_response = self._get_target_response(action)
        
        # Get LlamaGuard score
        llamaguard_score = self._get_llamaguard_score(target_response)
        
        # Extract embeddings
        action_embedding = self._extract_embedding(action, "action")
        response_embedding = self._extract_embedding(target_response, "response")
        context_embedding = self._extract_embedding(self._get_context_text(), "context")
        
        # Get policy scores
        v_high_score, mean_q_low_score = self._get_policy_scores(action, current_state)
        
        # Create turn
        turn = ConversationTurn(
            turn_number=turn_num,
            high_level_instruction=high_level_instruction,
            low_level_action=action,
            target_response=target_response,
            llamaguard_score=llamaguard_score,
            action_embedding=action_embedding,
            response_embedding=response_embedding,
            context_embedding=context_embedding,
            v_high_score=v_high_score,
            mean_q_low_score=mean_q_low_score
        )
        
        return turn
    
    def _generate_action(self, state: List[str], high_level_instruction: str) -> str:
        """Generate action using low-level policy."""
        # Convert state to input format for policy
        state_text = " ".join(state)
        
        # Use temperature variation if set, otherwise default
        temperature = getattr(self, '_current_generation_temperature', 0.7)
        
        # Generate action using policy's act method
        with torch.inference_mode():
            actions, action_tokens, log_probs = self.low_level_policy.act(
                state=[state_text],
                instructions=high_level_instruction,
                max_new_tokens=self.config.environment.max_new_tokens,
                temperature=temperature
            )
        
        # Return the first action
        return actions[0].strip()
    
    def _get_target_response(self, action: str) -> str:
        """Get target model response to action."""
        # Use environment to get target response
        # The method expects a list of chat histories (each history is a list of messages)
        chat_history = [{'role': 'user', 'content': action}]
        with torch.inference_mode():
            responses = self.environment._get_target_responses([chat_history])
        return responses[0]
    
    def _get_llamaguard_score(self, response: str) -> float:
        """Get LlamaGuard safety score for response."""
        with torch.inference_mode():
            scores = self.environment.score_safety([response], include_white_box=False)
        return float(scores[0])

    def _get_llamaguard_scores_batch(self, responses: List[str]) -> List[float]:
        """Get LlamaGuard safety scores for a batch of responses in one call."""
        if not responses:
            return []
        with torch.inference_mode():
            scores = self.environment.score_safety(responses, include_white_box=False)
        # Ensure CPU floats
        return [float(s.item() if torch.is_tensor(s) else float(s)) for s in scores]
    
    def _extract_embedding(self, text: str, embedding_type: str) -> torch.Tensor:
        """Extract embedding from target model."""
        # Use target model singleton for embedding extraction
        if self.singletons.get("target") is not None:
            tokenizer = self.singletons["target_tokenizer"]
            model = self.singletons["target"].base_model
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Use last hidden state mean
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states.mean(dim=1).squeeze()
                # Ensure float32 to avoid BFloat16 issues
                return embedding.float()
        else:
            # Use environment's embedding extraction method
            dummy_texts = ["dummy"]  # Single text for embedding
            embeddings = self.environment._get_embed_similarity_batched([text], dummy_texts)
            # Ensure float32 to avoid BFloat16 issues
            return embeddings[0].float()  # Return first embedding

    def _extract_embeddings_batch(self, texts: List[Any]) -> List[torch.Tensor]:
        """Extract embeddings for a batch of texts using a single forward pass."""
        # Flatten any nested lists to flat strings
        flat_texts = [" ".join(t) if isinstance(t, list) else str(t) for t in texts]
        if len(flat_texts) == 0:
            return []
        if self.singletons.get("target") is not None:
            tokenizer = self.singletons["target_tokenizer"]
            model = self.singletons["target"].base_model
            inputs = tokenizer(
                flat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # [batch, seq, dim]
                embs = hidden_states.mean(dim=1).float()   # [batch, dim]
            # Split into list of 1D tensors on CPU for downstream storage
            return [emb.detach().cpu() for emb in embs]
        else:
            # Fallback: mean-pool input embeddings (no transformer) similar to environment._get_embed_similarity_batched
            embedding_layer = self.environment.target_model.get_input_embeddings()
            tokenizer = self.environment.target_tokenizer
            device = embedding_layer.weight.device
            enc = tokenizer(
                flat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = enc.input_ids.long().to(device)
            attention_mask = enc.attention_mask.to(device)
            with torch.no_grad():
                token_embs = embedding_layer(input_ids)  # [B, T, D]
                masked = token_embs * attention_mask.unsqueeze(-1)
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                embs = (summed / counts).float()
            return [emb.detach().cpu() for emb in embs]
    
    def _get_context_text(self) -> str:
        """Get current conversation context as text."""
        conversation_histories = self.environment.get_agent_conversation_history()
        # Extract text content from conversation histories
        context_parts = []
        for history in conversation_histories:
            messages = [msg['content'] for msg in history]
            context_parts.extend(messages)
        return " ".join(context_parts)
    
    def _get_policy_scores(self, action: str, state: List[str]) -> Tuple[float, float]:
        """Get V_high and Q_low scores from hierarchical policies using trained critics."""
        # Fast path: if critics are not loaded, return zeros (fail fast elsewhere)
        if self.sequence_critic is None or self.token_critic is None or self.tokenizer is None:
            raise RuntimeError("Sequence and token critics must be loaded before computing policy scores")
        
        # Prepare concatenated text: state + action
        state_text = " ".join(state)
        combined_text = state_text + " " + action
        
        # Encode state and action separately (no specials) and concatenate for precise boundary
        enc_state = self.tokenizer(
            [state_text],
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        enc_action = self.tokenizer(
            [action],
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        state_ids = enc_state.input_ids
        action_ids = enc_action.input_ids
        state_len_full = state_ids.size(1)
        action_len_full = action_ids.size(1)
        max_len = 512
        # Ensure action tokens fit; trim state from the left if needed to keep full action
        keep_action_len = min(action_len_full, max_len)
        keep_state_len = max_len - keep_action_len
        if keep_state_len < state_len_full:
            state_ids = state_ids[:, -keep_state_len:] if keep_state_len > 0 else state_ids[:, :0]
        if keep_action_len < action_len_full:
            action_ids = action_ids[:, -keep_action_len:]
        # Concatenate
        input_ids = torch.cat([state_ids, action_ids], dim=1)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        # Use pad-aware attention mask to mirror AgentTrainer conventions
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -1
        attention_mask = (input_ids != pad_id).long()
        state_len = state_ids.size(1)
        
        # Place inputs on critic device for forward
        critic_device = next(self.sequence_critic.model.parameters()).device
        input_ids = input_ids.to(critic_device)
        attention_mask = attention_mask.to(critic_device)
        
        # Sanity checks: critics and shapes
        assert hasattr(self.sequence_critic, "is_token_level") and not self.sequence_critic.is_token_level, "sequence_critic must be sequence-level"
        assert hasattr(self.token_critic, "is_token_level") and self.token_critic.is_token_level, "token_critic must be token-level"
        seq_len = input_ids.size(1)
        assert attention_mask.size(1) == seq_len, "attention_mask length must equal input_ids length"
        
        # Sequence-level value for entire sequence (proxy for v_high)
        with torch.no_grad():
            v_high_val = self.sequence_critic(
                input_ids=input_ids,
                attention_mask=attention_mask,
                requires_grad=False,
            )
        # Expect scalar output per batch element
        assert v_high_val.dim() in (1, 2), "sequence_critic must return [batch] or [batch, 1]"
        v_high_score = float(v_high_val.squeeze().cpu().item())
        
        # Token-level values; take mean over action tokens only as mean q_low
        with torch.no_grad():
            token_values = self.token_critic(
                input_ids=input_ids,
                attention_mask=attention_mask,
                requires_grad=False,
            )  # [1, seq_len]
        # Expect per-token outputs
        assert token_values.dim() == 2 and token_values.size(1) == seq_len, "token_critic must return [batch, seq_len]"
        action_only_mask = attention_mask.clone().float()
        if state_len > 0:
            action_only_mask[:, :state_len] = 0.0
        # Verify we are averaging over exactly the action tokens
        expected_action_tokens = float(seq_len - state_len)
        assert abs(action_only_mask.sum().item() - expected_action_tokens) >= 0.0, "action_only_mask must cover only action tokens"
        denom = action_only_mask.sum(dim=1).clamp_min(1.0)
        mean_q_low = (token_values * action_only_mask).sum(dim=1) / denom
        mean_q_low_score = float(mean_q_low.squeeze().cpu().item())
        
        return v_high_score, mean_q_low_score

    def _get_policy_scores_batch(self, actions: List[str], states: List[Any]) -> Tuple[List[float], List[float]]:
        """Batched version of policy scoring for efficiency."""
        if self.sequence_critic is None or self.token_critic is None or self.tokenizer is None:
            raise RuntimeError("Sequence and token critics must be loaded before computing policy scores")

        batch_size = len(actions)
        assert batch_size == len(states), "actions and states batch size mismatch"

        # Flatten any nested state lists to flat strings
        state_texts = [" ".join(s) if isinstance(s, list) else str(s) for s in states]
        enc_state = self.tokenizer(
            state_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        enc_action = self.tokenizer(
            actions,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=False
        )

        state_ids_list: List[torch.Tensor] = [row for row in enc_state.input_ids]
        action_ids_list: List[torch.Tensor] = [row for row in enc_action.input_ids]

        max_len = 512
        per_sample_input_ids: List[torch.Tensor] = []
        per_sample_state_lens: List[int] = []
        for st_ids, ac_ids in zip(state_ids_list, action_ids_list):
            state_len_full = int(st_ids.size(0))
            action_len_full = int(ac_ids.size(0))
            keep_action_len = min(action_len_full, max_len)
            keep_state_len = max_len - keep_action_len
            st_trim = st_ids[-keep_state_len:] if keep_state_len > 0 and state_len_full > keep_state_len else st_ids[:0] if keep_state_len <= 0 else st_ids
            ac_trim = ac_ids[-keep_action_len:] if keep_action_len < action_len_full else ac_ids
            sample_ids = torch.cat([st_trim, ac_trim], dim=0)
            per_sample_input_ids.append(sample_ids)
            per_sample_state_lens.append(int(st_trim.size(0)))

        # Pad to a rectangular batch
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        # Import left_pad_sequence for proper left-padding of decoder-only models
        from src.mcq_agent_trainer import left_pad_sequence
        padded_input_ids = left_pad_sequence(per_sample_input_ids, batch_first=True, padding_value=pad_id)
        if padded_input_ids.dtype != torch.long:
            padded_input_ids = padded_input_ids.long()
        attention_mask = (padded_input_ids != pad_id).long()

        # Move to critic device
        critic_device = next(self.sequence_critic.model.parameters()).device
        padded_input_ids = padded_input_ids.to(critic_device)
        attention_mask = attention_mask.to(critic_device)

        # Sequence-level values
        with torch.no_grad():
            v_high_batch = self.sequence_critic(
                input_ids=padded_input_ids,
                attention_mask=attention_mask,
                requires_grad=False,
            )
        if v_high_batch.dim() == 2 and v_high_batch.size(1) == 1:
            v_high_batch = v_high_batch.squeeze(1)
        v_high_scores = [float(x) for x in v_high_batch.detach().cpu().tolist()]

        # Token-level values (mean over action tokens only)
        with torch.no_grad():
            token_values = self.token_critic(
                input_ids=padded_input_ids,
                attention_mask=attention_mask,
                requires_grad=False,
            )  # [B, T]
        assert token_values.dim() == 2 and token_values.size(0) == batch_size
        max_seq_len = token_values.size(1)
        action_masks = attention_mask.clone().float()
        for i, st_len in enumerate(per_sample_state_lens):
            if st_len > 0:
                action_masks[i, :st_len] = 0.0
        denom = action_masks.sum(dim=1).clamp_min(1.0)
        mean_q_low = (token_values * action_masks).sum(dim=1) / denom
        mean_q_low_scores = [float(x) for x in mean_q_low.detach().cpu().tolist()]

        return v_high_scores, mean_q_low_scores
    
    def _save_intermediate_results(self, conversations: List[SemanticConversation], metadata: Dict[str, Any]):
        """Save intermediate results during collection."""
        timestamp = int(time.time())
        
        # Save conversations
        for conversation in conversations[-self.config.semantic_analysis.save_interval:]:
            conv_path = os.path.join(
                self.config.semantic_analysis.output_dir,
                "conversations",
                f"{conversation.conversation_id}.json"
            )
            conversation.save_to_json(conv_path)
        
        # Save metadata
        metadata_path = os.path.join(
            self.config.semantic_analysis.output_dir,
            "metadata",
            f"metadata_{timestamp}.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save vector store (autosave)
        vector_path = os.path.join(
            self.config.semantic_analysis.output_dir,
            "vectors"
        )
        try:
            self.vector_store.save(vector_path)
        except Exception as e:
            raise RuntimeError(f"Failed to autosave vector store: {e}")
        
        print(f"Saved intermediate results: {len(conversations)} conversations and vector store autosaved.")
    
    def _save_final_results(self, conversations: List[SemanticConversation], metadata: Dict[str, Any]):
        """Save final results and vector store."""
        # Save all conversations
        for conversation in conversations:
            conv_path = os.path.join(
                self.config.semantic_analysis.output_dir,
                "conversations",
                f"{conversation.conversation_id}.json"
            )
            conversation.save_to_json(conv_path)
        
        # Save vector store
        vector_path = os.path.join(
            self.config.semantic_analysis.output_dir,
            "vectors"
        )
        self.vector_store.save(vector_path)
        
        # Save final metadata
        metadata_path = os.path.join(
            self.config.semantic_analysis.output_dir,
            "metadata",
            "final_metadata.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save statistics
        stats = self.vector_store.get_statistics()
        stats.update(metadata)
        stats_path = os.path.join(
            self.config.semantic_analysis.output_dir,
            "metadata",
            "statistics.json"
        )
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved final results: {len(conversations)} conversations")
        print(f"Vector store statistics: {stats}") 

    def _process_conversation_batch(self, questions: List[str], categories: List[str], variant_ids: List[int]) -> List[Optional[SemanticConversation]]:
        """Process multiple conversations concurrently using batched environment operations."""
        batch_size = len(questions)
        conversations = []
        
        # Set up temperature variations for each conversation in the batch
        import random
        base_temperature = 0.7
        temperatures = []
        for _ in range(batch_size):
            temperature_variation = random.uniform(-0.1, 0.1)
            generation_temperature = max(0.1, min(1.0, base_temperature + temperature_variation))
            temperatures.append(generation_temperature)
        
        # Create conversation IDs
        conversation_ids = []
        for i, variant_id in enumerate(variant_ids):
            conversation_id = f"conv_{self.conversation_id_counter:06d}_v{variant_id}"
            conversation_ids.append(conversation_id)
            self.conversation_id_counter += 1
        
        # Set environment state for batch conversations
        # The environment expects a list of conversation histories
        initial_states = [[question] for question in questions]
        self.environment.set_state(initial_states)
        
        # Initialize conversation data structures
        batch_turns = [[] for _ in range(batch_size)]
        max_turns = self.config.semantic_analysis.max_conversation_length
        
        # Process turns for the entire batch simultaneously
        for turn_num in range(max_turns):
            # Store current temperatures for this turn
            for i, temp in enumerate(temperatures):
                self._current_generation_temperature = temp
                
            try:
                # Get current conversation states
                current_states = self.environment.get_agent_conversation_history()
                
                # Process this turn for all conversations in the batch
                batch_turn_data = self._process_batch_turn(
                    turn_num, questions, current_states, temperatures
                )
                
                # Store turn data for each conversation
                for i, turn_data in enumerate(batch_turn_data):
                    if turn_data:
                        batch_turns[i].append(turn_data)
                        
            except Exception as e:
                print(f"Error processing batch turn {turn_num}: {e}")
                raise e
        
        # Clean up temperature override
        if hasattr(self, '_current_generation_temperature'):
            delattr(self, '_current_generation_temperature')
        
        # Create conversation objects from collected turns
        for i in range(batch_size):
            if batch_turns[i]:  # Only create if we have turns
                try:
                    # Compute running max harm R_{0:t} for each conversation
                    max_lg = float('-inf')
                    max_vh = float('-inf')
                    max_ql = float('-inf')
                    for t in batch_turns[i]:
                        max_lg = max(max_lg, t.llamaguard_score)
                        max_vh = max(max_vh, t.v_high_score)
                        max_ql = max(max_ql, t.mean_q_low_score)
                        t.max_llamaguard_to_t = max_lg
                        t.max_v_high_to_t = max_vh
                        t.max_q_low_to_t = max_ql
                    
                    # Generate original prompt embedding for the question
                    original_prompt_embedding = self._extract_embedding(questions[i], "context")
                    
                    conversation = SemanticConversation(
                        conversation_id=conversation_ids[i],
                        original_question=questions[i],
                        category=categories[i],
                        turns=batch_turns[i],
                        original_prompt_embedding=original_prompt_embedding
                    )
                    
                    # Add to vector store
                    embedding_arrays = conversation.get_embedding_arrays()
                    self.vector_store.add_conversation_vectors(
                        conversation_id=conversation_ids[i],
                        action_embeddings=embedding_arrays['action_embeddings'],
                        response_embeddings=embedding_arrays['response_embeddings'],
                        context_embeddings=embedding_arrays['context_embeddings']
                    )
                    
                    conversations.append(conversation)
                except Exception as e:
                    print(f"Error creating conversation {i}: {e}")
                    raise e
            else:
                conversations.append(None)
        
        return conversations

    def _process_batch_turn(self, turn_num: int, original_questions: List[str], current_states: List[List[str]], temperatures: List[float]) -> List[Optional[ConversationTurn]]:
        """Process a single turn for a batch of conversations."""
        batch_size = len(original_questions)
        turn_data = []
        
        try:
            # Prepare state texts for batch tokenization
            state_texts = [
                self._get_context_text_from_state(current_states[i])
                for i in range(batch_size)
            ]
            enc = self.high_level_policy.tokenizer(
                state_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask
            high_level_instructions = [
                self.high_level_policy.select_action(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0))
                for i in range(batch_size)
            ]
            
            # Generate low-level actions for the batch
            actions = []
            for i in range(batch_size):
                # Set temperature for this conversation
                self._current_generation_temperature = temperatures[i]
                
                # Extract text content from conversation history
                current_state = []
                for msg in current_states[i]:
                    if isinstance(msg, dict) and 'content' in msg:
                        current_state.append(msg['content'])
                    else:
                        current_state.append(str(msg))
                
                action = self._generate_action(current_state, high_level_instructions[i])
                actions.append(action)
            
            # Get target responses using environment step
            # Note: This updates the environment state automatically
            with torch.inference_mode():
                step_result = self.environment.step(actions, return_generation_hidden_states=True)
                if isinstance(step_result, tuple) and len(step_result) == 5:
                    next_states, target_responses, _, _, _gen_hidden = step_result
                else:
                    next_states, target_responses, _, _ = step_result
                    _gen_hidden = None
            
            # Batch compute LlamaGuard scores
            lg_scores = self._get_llamaguard_scores_batch(target_responses)

            # Prepare context texts for embedding
            context_texts = [self._get_context_text_from_state(cs) for cs in current_states]

            # Batch compute embeddings
            action_embeddings = self._extract_embeddings_batch(actions)
            response_embeddings = self._extract_embeddings_batch(target_responses)
            context_embeddings = self._extract_embeddings_batch(context_texts)

            # Batch compute critic scores
            # Build state texts for critic scoring
            states_for_critics: List[List[str]] = []
            for cs in current_states:
                st = []
                for msg in cs:
                    if isinstance(msg, dict) and 'content' in msg:
                        st.append(msg['content'])
                    else:
                        st.append(str(msg))
                states_for_critics.append(st)
            v_high_scores, q_low_scores = self._get_policy_scores_batch(actions, states_for_critics)

            # Assemble per-sample turns
            for i in range(batch_size):
                try:
                    turn = ConversationTurn(
                        turn_number=turn_num,
                        high_level_instruction=high_level_instructions[i],
                        low_level_action=actions[i],
                        target_response=target_responses[i],
                        llamaguard_score=float(lg_scores[i]),
                        action_embedding=action_embeddings[i],
                        response_embedding=response_embeddings[i],
                        context_embedding=context_embeddings[i],
                        v_high_score=float(v_high_scores[i]),
                        mean_q_low_score=float(q_low_scores[i])
                    )
                    turn_data.append(turn)
                except Exception as e:
                    raise e
            
        except Exception as e:
            print(f"Error in batch turn processing: {e}")
            raise e
        
        return turn_data

    def _get_context_text_from_state(self, conversation_state: List[str]) -> str:
        """Get context text from a specific conversation state."""
        context_parts = []
        for msg in conversation_state:
            if isinstance(msg, dict) and 'content' in msg:
                context_parts.append(msg['content'])
            else:
                context_parts.append(str(msg))
        return " ".join(context_parts) 