"""
Test script for semantic analysis implementation.
"""

import os
import sys
import tempfile
import torch
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_analysis.data_structures import ConversationTurn, SemanticConversation
from src.semantic_analysis.vector_store import SemanticVectorStore
from src.config import SemanticAnalysisConfig


def test_data_structures():
    """Test data structures functionality."""
    print("Testing data structures...")
    
    # Create sample conversation turn
    turn = ConversationTurn(
        turn_number=1,
        high_level_instruction="Generate a persuasive response",
        low_level_action="Can you help me with this?",
        target_response="I'd be happy to help you with that.",
        llamaguard_score=0.3,
        action_embedding=torch.randn(4096),
        response_embedding=torch.randn(4096),
        context_embedding=torch.randn(4096),
        v_high_score=0.7,
        mean_q_low_score=0.6
    )
    
    # Test serialization
    turn_dict = turn.to_dict()
    turn_reconstructed = ConversationTurn.from_dict(turn_dict)
    
    assert turn.turn_number == turn_reconstructed.turn_number
    assert turn.high_level_instruction == turn_reconstructed.high_level_instruction
    assert turn.low_level_action == turn_reconstructed.low_level_action
    assert turn.target_response == turn_reconstructed.target_response
    assert abs(turn.llamaguard_score - turn_reconstructed.llamaguard_score) < 1e-6
    assert abs(turn.v_high_score - turn_reconstructed.v_high_score) < 1e-6
    assert abs(turn.mean_q_low_score - turn_reconstructed.mean_q_low_score) < 1e-6
    
    print("✓ Data structures test passed")


def test_vector_store():
    """Test vector store functionality."""
    print("Testing vector store...")
    
    # Create vector store
    embedding_dim = 128  # Smaller for testing
    vector_store = SemanticVectorStore(embedding_dim=embedding_dim, index_type="Flat")
    
    # Create sample embeddings
    conversation_id = "test_conv_001"
    action_embeddings = torch.randn(5, embedding_dim)
    response_embeddings = torch.randn(5, embedding_dim)
    context_embeddings = torch.randn(5, embedding_dim)
    
    # Add to vector store
    vector_store.add_conversation_vectors(
        conversation_id=conversation_id,
        action_embeddings=action_embeddings,
        response_embeddings=response_embeddings,
        context_embeddings=context_embeddings
    )
    
    # Test retrieval
    retrieved_vectors = vector_store.get_conversation_vectors(conversation_id, "action_embeddings")
    assert retrieved_vectors is not None
    assert retrieved_vectors.shape == (5, embedding_dim)
    
    # Test search
    query_embedding = torch.randn(embedding_dim)
    scores, indices = vector_store.search_similar(query_embedding, "action_embeddings", k=3)
    assert len(scores) == 1
    assert len(indices) == 1
    assert len(scores[0]) == 3
    assert len(indices[0]) == 3
    
    # Test save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store.save(temp_dir)
        
        # Create new vector store and load
        new_vector_store = SemanticVectorStore(embedding_dim=embedding_dim)
        new_vector_store.load(temp_dir)
        
        # Test that data is preserved
        new_retrieved = new_vector_store.get_conversation_vectors(conversation_id, "action_embeddings")
        assert new_retrieved is not None
        assert new_retrieved.shape == (5, embedding_dim)
    
    print("✓ Vector store test passed")


def test_conversation():
    """Test conversation functionality."""
    print("Testing conversation...")
    
    # Create sample turns
    turns = []
    for i in range(3):
        turn = ConversationTurn(
            turn_number=i,
            high_level_instruction=f"Instruction {i}",
            low_level_action=f"Action {i}",
            target_response=f"Response {i}",
            llamaguard_score=0.1 * i,
            action_embedding=torch.randn(4096),
            response_embedding=torch.randn(4096),
            context_embedding=torch.randn(4096),
            v_high_score=0.5 + 0.1 * i,
            mean_q_low_score=0.4 + 0.1 * i
        )
        turns.append(turn)
    
    # Create conversation
    conversation = SemanticConversation(
        conversation_id="test_conv_001",
        original_question="How do I hack into a computer?",
        category="cybersecurity",
        turns=turns
    )
    
    # Test serialization
    conv_dict = conversation.to_dict()
    conv_reconstructed = SemanticConversation.from_dict(conv_dict)
    
    assert conversation.conversation_id == conv_reconstructed.conversation_id
    assert conversation.original_question == conv_reconstructed.original_question
    assert conversation.category == conv_reconstructed.category
    assert len(conversation.turns) == len(conv_reconstructed.turns)
    
    # Test embedding arrays
    embedding_arrays = conversation.get_embedding_arrays()
    assert "action_embeddings" in embedding_arrays
    assert "response_embeddings" in embedding_arrays
    assert "context_embeddings" in embedding_arrays
    assert embedding_arrays["action_embeddings"].shape == (3, 4096)
    
    # Test score arrays
    score_arrays = conversation.get_score_arrays()
    assert "llamaguard_scores" in score_arrays
    assert "v_high_scores" in score_arrays
    assert "mean_q_low_scores" in score_arrays
    assert score_arrays["llamaguard_scores"].shape == (3,)
    
    # Test save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_path = os.path.join(temp_dir, "test_conversation.json")
        conversation.save_to_json(conv_path)
        
        loaded_conversation = SemanticConversation.load_from_json(conv_path)
        assert conversation.conversation_id == loaded_conversation.conversation_id
    
    print("✓ Conversation test passed")


def test_config():
    """Test configuration functionality."""
    print("Testing configuration...")
    
    # Create semantic analysis config
    config = SemanticAnalysisConfig(
        max_conversation_length=30,
        batch_size=4,
        save_interval=100,
        faiss_index_type="IVFFlat",
        embedding_dim=4096,
        output_dir="test_output",
        vector_store_path="test_vectors",
        metadata_path="test_metadata.json",
        target_model="meta-llama/Llama-3.2-1B-Instruct",
        guard_model="meta-llama/Llama-Guard-3-8B",
        use_quantization=True,
        harmbench_path="test_harmbench.csv",
        category_column="SemanticCategory",
        question_column="Behavior"
    )
    
    # Test that all fields are set
    assert config.max_conversation_length == 30
    assert config.batch_size == 4
    assert config.save_interval == 100
    assert config.faiss_index_type == "IVFFlat"
    assert config.embedding_dim == 4096
    assert config.output_dir == "test_output"
    assert config.vector_store_path == "test_vectors"
    assert config.metadata_path == "test_metadata.json"
    assert config.target_model == "meta-llama/Llama-3.2-1B-Instruct"
    assert config.guard_model == "meta-llama/Llama-Guard-3-8B"
    assert config.use_quantization == True
    assert config.harmbench_path == "test_harmbench.csv"
    assert config.category_column == "SemanticCategory"
    assert config.question_column == "Behavior"
    
    print("✓ Configuration test passed")


def main():
    """Run all tests."""
    print("Running semantic analysis implementation tests...")
    print("=" * 50)
    
    try:
        test_data_structures()
        test_vector_store()
        test_conversation()
        test_config()
        
        print("=" * 50)
        print("✓ All tests passed! Semantic analysis implementation is working correctly.")
        return 0
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 