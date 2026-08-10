"""
Recompute v_high, q_low, LlamaGuard, and optionally embeddings for saved conversations.
Overwrites the conversation JSON files and the FAISS vector store.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the parent directory to the path so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from src.base import BaseModelSingleton
from src.semantic_analysis.data_structures import SemanticConversation, ConversationTurn
from src.semantic_analysis.vector_store import SemanticVectorStore
from src.critics import SequenceLevelValueCritic, TokenLevelValueCritic


def _load_singletons_and_models(checkpoint_dir: str, env_config: Any = None) -> Dict[str, Any]:
    """Load shared singletons and critics needed for recomputation.
    Expects the same model used during collection (agent/critic backbones) and guard model via environment config.
    """
    # Device preferences similar to collector
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    agent_prefer_gpu = 0 if num_gpus > 0 else None
    target_prefer_gpu = 1 if num_gpus > 1 else 0 if num_gpus == 1 else None
    guard_prefer_gpu = 1 if num_gpus > 1 else 0 if num_gpus == 1 else None

    # Minimal config handling: get model names from saved config in checkpoint
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in checkpoint: {checkpoint_dir}")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # Extract required fields
    hf_checkpoint = cfg["model"]["hf_checkpoint"]
    target_model = cfg["environment"].get("target_model")
    guard_model = cfg["environment"].get("guard_model")
    target_quantize = cfg["environment"].get("target_quantize")
    guard_quantize = cfg["environment"].get("guard_quantize")

    # Token
    hf_token = os.environ.get("HF_TOKEN") or None

    # Agent/Critic singleton (use same backbone for critics)
    agent_singleton = BaseModelSingleton.get_instance(
        model_name=hf_checkpoint,
        hf_token=hf_token,
        use_offloading=False,
        config=None,
        prefer_gpu=agent_prefer_gpu,
        quantize=False,
    )

    # Tokenizer from hf_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, use_fast=True, padding_side='left')
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})

    # Load critics
    seq_dir = os.path.join(checkpoint_dir, "sequence_critic")
    tok_dir = os.path.join(checkpoint_dir, "token_critic")
    if not os.path.exists(seq_dir) or not os.path.exists(tok_dir):
        raise FileNotFoundError("sequence_critic and token_critic not found in checkpoint")
    # Fail-fast verification of expected value head files
    seq_head_path = os.path.join(seq_dir, "sequence_value_head.pt")
    tok_head_path = os.path.join(tok_dir, "token_value_head.pt")
    if not os.path.exists(seq_head_path):
        raise FileNotFoundError(f"Missing sequence critic value head: {seq_head_path}")
    if not os.path.exists(tok_head_path):
        raise FileNotFoundError(f"Missing token critic value head: {tok_head_path}")

    shared_backbone = agent_singleton.shared_model
    sequence_critic = SequenceLevelValueCritic(hf_checkpoint=hf_checkpoint, tokenizer=tokenizer, model=shared_backbone)
    sequence_critic.load(seq_dir)
    token_critic = TokenLevelValueCritic(hf_checkpoint=hf_checkpoint, tokenizer=tokenizer, model=shared_backbone)
    token_critic.load(tok_dir)

    # Guard singleton (optional; if absent we will skip LlamaGuard recompute)
    guard_singleton = None
    guard_tokenizer = None
    if guard_model:
        guard_tokenizer = AutoTokenizer.from_pretrained(guard_model, use_fast=True, padding_side='left')
        guard_tokenizer.padding_side = 'left'
        if guard_tokenizer.pad_token is None:
            if guard_tokenizer.eos_token is not None:
                guard_tokenizer.pad_token = guard_tokenizer.eos_token
                guard_tokenizer.pad_token_id = guard_tokenizer.eos_token_id
            else:
                guard_tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
        guard_base = AutoModelForCausalLM.from_pretrained(
            guard_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        guard_base = guard_base.to('cuda:1' if torch.cuda.device_count() > 1 else 'cuda')
        class _Wrap:
            def __init__(self, base_model):
                self.base_model = base_model
        guard_singleton = _Wrap(guard_base)

    return {
        "tokenizer": tokenizer,
        "sequence_critic": sequence_critic,
        "token_critic": token_critic,
        # Use the critic backbone for embedding extraction to ensure consistency
        "embedding_model": shared_backbone,
        "guard_model": guard_singleton.base_model if guard_singleton else None,
        "guard_tokenizer": guard_tokenizer,
        "hf_checkpoint": hf_checkpoint,
        "target_model": target_model,
    }


def _compute_scores_for_turn(state_text: str, action: str, target_response: str, models: Dict[str, Any]) -> Dict[str, float]:
    """Compute v_high, mean q_low (using state context), and LlamaGuard score for a single turn."""
    tokenizer = models["tokenizer"]
    sequence_critic = models["sequence_critic"]
    token_critic = models["token_critic"]
    guard_model = models.get("guard_model")
    guard_tokenizer = models.get("guard_tokenizer")

    # Build state+action without special tokens and ensure action tokens preserved fully
    enc_state = tokenizer([state_text], add_special_tokens=False, return_tensors="pt", truncation=True, max_length=512)
    enc_action = tokenizer([action], add_special_tokens=False, return_tensors="pt", truncation=True, max_length=512)
    # Place ids on the same device as critics for forward
    critic_device = next(sequence_critic.model.parameters()).device
    state_ids = enc_state.input_ids.to(critic_device)
    action_ids = enc_action.input_ids.to(critic_device)
    state_len_full = state_ids.size(1)
    action_len_full = action_ids.size(1)
    max_len = 512
    keep_action_len = min(action_len_full, max_len)
    keep_state_len = max_len - keep_action_len
    if keep_state_len < state_len_full:
        state_ids = state_ids[:, -keep_state_len:] if keep_state_len > 0 else state_ids[:, :0]
    if keep_action_len < action_len_full:
        action_ids = action_ids[:, -keep_action_len:]
    input_ids = torch.cat([state_ids, action_ids], dim=1)
    # Ensure correct dtype for embedding indices
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()
    # Use pad-aware attention mask to mirror AgentTrainer conventions
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    attention_mask = (input_ids != pad_id).long()
    state_len = state_ids.size(1)

    with torch.no_grad():
        v_high_val = sequence_critic(input_ids=input_ids, attention_mask=attention_mask, requires_grad=False)
        token_values = token_critic(input_ids=input_ids, attention_mask=attention_mask, requires_grad=False)
    # Sanity checks on outputs
    assert v_high_val.dim() in (1, 2), "sequence_critic must return [batch] or [batch, 1]"
    assert token_values.dim() == 2 and token_values.size(1) == input_ids.size(1), "token_critic must return [batch, seq_len]"
    v_high_score = float(v_high_val.squeeze().cpu().item())
    action_only_mask = attention_mask.clone().float().to(token_values.device)
    if state_len > 0:
        action_only_mask[:, :state_len] = 0.0
    # Verify averaging over action tokens only
    expected_action_tokens = float(input_ids.size(1) - state_len)
    assert action_only_mask.sum().item() >= 0.0, "action_only_mask must be valid"
    denom = action_only_mask.sum(dim=1).clamp_min(1.0)
    mean_q_low = (token_values * action_only_mask).sum(dim=1) / denom
    mean_q_low_score = float(mean_q_low.squeeze().cpu().item())

    # LlamaGuard harmfulness (optional)
    llamaguard_score = None
    if guard_model is not None and guard_tokenizer is not None:
        # Following environment.score_safety: 1 - P(safe) at position 1
        chat = [[{'role': 'user', 'content': target_response[-2000:]}]]
        batch_inputs = guard_tokenizer.apply_chat_template(chat, return_tensors='pt', return_dict=True, padding=True, padding_side='left', tokenize=True)
        # Route to guard model's embedding device to support sharded models
        guard_dev = guard_model.get_input_embeddings().weight.device if hasattr(guard_model, 'get_input_embeddings') else next(guard_model.parameters()).device
        input_ids = batch_inputs.input_ids.to(guard_dev)
        attention_mask = batch_inputs.attention_mask.to(guard_dev)
        
        outputs = guard_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=5, pad_token_id=guard_tokenizer.eos_token_id, return_dict_in_generate=True, output_logits=True)
        logits = torch.stack([logit for logit in outputs.logits], dim=1)
        probs = torch.softmax(logits, dim=-1)
        # Ensure we read the correct token id for 'safe' on the guard tokenizer
        safe_id = guard_tokenizer.convert_tokens_to_ids('safe')
        llamaguard_score = float((1 - torch.clamp(probs[:, 1, safe_id], 0.0, 1.0)).squeeze().cpu().item())

    return {
        "v_high_score": v_high_score,
        "mean_q_low_score": mean_q_low_score,
        "llamaguard_score": llamaguard_score,
    }


def _extract_embeddings_for_texts(texts: List[str], tokenizer: Any, model: Any) -> List[torch.Tensor]:
    """Extract mean-pooled last hidden-state embeddings for a batch of texts."""
    if len(texts) == 0:
        return []
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        embs = hidden.mean(dim=1).float()
    # Return CPU tensors
    return [e.detach().cpu() for e in embs]


def recompute_scores_for_dataset(data_dir: str, checkpoint_dir: str, recompute_embeddings: bool = False) -> None:
    """Recompute scores for all conversations in data_dir/conversations and overwrite JSON and vector store.

    If recompute_embeddings is True, also regenerate action/response/context embeddings from text and overwrite them in the
    conversation files before rebuilding the vector store.
    """
    conv_dir = os.path.join(data_dir, "conversations")
    if not os.path.isdir(conv_dir):
        raise FileNotFoundError(f"Conversations directory not found: {conv_dir}")

    # Load models
    models = _load_singletons_and_models(checkpoint_dir)
    tokenizer = models["tokenizer"]
    embedding_model = models["embedding_model"]

    # List conversation files
    files = [f for f in os.listdir(conv_dir) if f.endswith('.json')]
    if not files:
        raise RuntimeError(f"No conversation JSON files found in {conv_dir}")

    # Determine embedding dimension from the first conversation's stored embeddings
    first_path = os.path.join(conv_dir, files[0])
    first_conv = SemanticConversation.load_from_json(first_path)
    if not first_conv.turns:
        raise RuntimeError(f"First conversation has no turns: {files[0]}")
    emb_dim = int(first_conv.turns[0].context_embedding.shape[0])

    # Prepare vector store (rebuild from scratch) with correct embedding dimension
    vectors_path = os.path.join(data_dir, "vectors")
    vector_store = SemanticVectorStore(embedding_dim=emb_dim, index_type="IVFFlat")

    # Iterate conversations
    for fname in files:
        fpath = os.path.join(conv_dir, fname)
        conv = SemanticConversation.load_from_json(fpath)

        # Build state context incrementally from stored turns
        turns: List[ConversationTurn] = []
        running_max_lg = float('-inf')
        running_max_vh = float('-inf')
        running_max_ql = float('-inf')

        # Reconstruct state text per turn from conversation data (previous messages)
        for turn in conv.turns:
            # Use previous context + previous model responses + instructions (if present in turn)
            state_text = conv.original_question + " " + " ".join(
                [t.low_level_action + " " + t.target_response for t in turns]
            ).strip()

            scores = _compute_scores_for_turn(state_text=state_text, action=turn.low_level_action, target_response=turn.target_response, models=models)

            # Update running max
            if scores["llamaguard_score"] is not None:
                running_max_lg = max(running_max_lg, scores["llamaguard_score"]) if running_max_lg != float('-inf') else scores["llamaguard_score"]
            running_max_vh = max(running_max_vh, scores["v_high_score"]) if running_max_vh != float('-inf') else scores["v_high_score"]
            running_max_ql = max(running_max_ql, scores["mean_q_low_score"]) if running_max_ql != float('-inf') else scores["mean_q_low_score"]

            # Overwrite scores on turn
            turn.llamaguard_score = scores["llamaguard_score"] if scores["llamaguard_score"] is not None else turn.llamaguard_score
            turn.v_high_score = scores["v_high_score"]
            turn.mean_q_low_score = scores["mean_q_low_score"]
            turn.max_llamaguard_to_t = running_max_lg if scores["llamaguard_score"] is not None else turn.max_llamaguard_to_t
            turn.max_v_high_to_t = running_max_vh
            turn.max_q_low_to_t = running_max_ql

            # Optionally recompute embeddings for this turn
            if recompute_embeddings:
                action_text = turn.low_level_action
                response_text = turn.target_response
                # Context text defined as prior context plus current pair
                context_text = (state_text + " " + action_text + " " + response_text).strip()
                a_emb, r_emb, c_emb = _extract_embeddings_for_texts([action_text, response_text, context_text], tokenizer, embedding_model)
                turn.action_embedding = a_emb
                turn.response_embedding = r_emb
                turn.context_embedding = c_emb

            turns.append(turn)

        # Save updated conversation
        conv.turns = turns
        conv.save_to_json(fpath)

        # Rebuild vector store entries for this conversation using stored embeddings
        embedding_arrays = conv.get_embedding_arrays()
        vector_store.add_conversation_vectors(
            conversation_id=conv.conversation_id,
            action_embeddings=embedding_arrays['action_embeddings'],
            response_embeddings=embedding_arrays['response_embeddings'],
            context_embeddings=embedding_arrays['context_embeddings']
        )

    # Save vector store (overwrite)
    vector_store.save(vectors_path) 


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute semantic analysis scores and optionally embeddings from saved conversations.")
    parser.add_argument("--data_dir", required=True, help="Directory containing conversations/ and vectors/")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory with config.json and critic heads")
    parser.add_argument("--recompute_embeddings", action="store_true", help="Also recompute embeddings from text and overwrite in JSON")
    args = parser.parse_args()

    recompute_scores_for_dataset(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir, recompute_embeddings=args.recompute_embeddings)
    print("Recompute complete.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)