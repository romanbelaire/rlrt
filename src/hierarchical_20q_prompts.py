"""
System prompts and prompt formatting for hierarchical 20 Questions agents.

This module provides the system prompts and context formatting functions
for both high-level (meta-cognitive) and low-level (question generation) 
agents in the hierarchical 20 Questions architecture.
"""

from typing import Dict, Any, Optional, List, Tuple


def get_high_level_system_prompt() -> str:
    """
    Get the system prompt for the high-level meta-cognitive agent.
    
    Returns:
        System prompt string for the high-level policy
    """
    return """You are one member of a pair of agents playing 20 questions as a team. Your role is to provide strategic guidance to your partner, who will ask the actual questions to the oracle.

Your partner is trying to guess a mystery word by asking yes/no questions. You have a maximum of 20 questions total to identify the word correctly.

IMPORTANT: You do NOT ask questions directly. Instead, you provide strategic instructions to guide your partner's questioning approach.

Your responsibilities:
1. Analyze the conversation history and current information state
2. Assess what has been learned and what gaps remain
3. Choose the most effective questioning strategy for the current situation
4. Provide clear, specific guidance to your partner

Available strategic approaches:
1. **Ask a question to gain more information** - When you need to gather additional data
   - Focus on broad categories (living vs non-living, natural vs artificial)
   - Explore specific properties (size, color, material, function)
   - Use elimination strategies to rule out possibilities
   - Clarify contradictions or ambiguous previous answers

2. **Make a final guess about the mystery word** - When you have sufficient information
   - You're confident about the word based on gathered evidence
   - You're running out of questions and need to make your best guess
   - The pattern of answers strongly points to a specific word

Your output should be a single, clear instruction to your partner. Examples:

For gathering information:
- "Ask about the broad category - determine if this is a living thing or not."
- "Focus on size - find out if this object is larger than a car."
- "Explore the material properties - ask what this item is made of."
- "Use elimination - ask about a specific category we haven't ruled out yet."
- "Clarify the contradiction between the previous answers about location."

For making a guess:
- "Make your final guess - based on the clues, I believe the word is 'elephant'."
- "Time to guess - the evidence strongly suggests this is 'basketball'."
- "Make your best guess now - we have enough information and limited questions left."

Keep your instructions concise, specific, and actionable. Focus on the strategic reasoning behind your choice."""


def get_low_level_system_prompt() -> str:
    """
    Get the system prompt for the low-level question generation agent.
    
    Returns:
        System prompt string for the low-level policy
    """
    return """You are playing 20 questions to guess a mystery word. You can ask yes/no questions and the oracle will respond with "Yes" or "No".

Your goal is to identify the mystery word within 20 questions by asking strategic yes/no questions.

Rules:
- You can only ask yes/no questions (questions that can be answered with "Yes" or "No")
- You have a maximum of 20 questions total to guess the word
- When making a final guess, phrase it as a yes/no question (e.g., "Is it an elephant?")
- Pay attention to previous questions and answers to avoid repetition
- Use the strategic advice provided to guide your questioning approach

You will receive strategic advice to help guide your questions. Use this advice to formulate effective yes/no questions that follow the suggested approach.

Examples of good questions:
- "Is it a living thing?"
- "Is it larger than a car?"
- "Is it made of metal?"
- "Is it an elephant?" (for final guesses)

Keep your questions clear, direct, and focused on gathering the most useful information possible."""


def format_high_level_context(
    conversation_context: str,
    information_summary: Dict[str, Any]
) -> str:
    """
    Format the context for the high-level meta-cognitive agent.
    
    Args:
        conversation_context: Formatted conversation history
        information_summary: Summary of information gathered
        
    Returns:
        Formatted context string for high-level policy
    """
    context_parts = [
        "CURRENT GAME STATE:",
        conversation_context,
        "",
        "Provide your strategic instruction to your partner:"
    ]
    
    return "\n".join(context_parts)


def format_low_level_context(
    hl_instruction: str,
    conversation_context: str,
    game_status: str
) -> str:
    """
    Format the context for the low-level question generation agent.
    
    Args:
        hl_instruction: Strategic instruction from high-level agent
        conversation_context: Formatted conversation history  
        game_status: Current game status information
        
    Returns:
        Formatted context string for low-level policy
    """
    context_parts = [
        f"STRATEGIC GUIDANCE: {hl_instruction}",
        "",
        f"GAME STATUS: {game_status}",
        "",
        "CONVERSATION HISTORY:",
        conversation_context,
        "",
        "Based on the strategic guidance above, provide your response:"
    ]
    
    return "\n".join(context_parts)


def format_low_level_oracle_context(
    questions_asked: List[str],
    oracle_answers: List[str],
    hl_instruction: str,
    question_count: int,
    max_questions: int = 20
) -> str:
    """
    Format the context for the low-level agent from the oracle's perspective.
    
    This presents the game as if the oracle is speaking directly to the LL agent,
    avoiding the three-way conversation confusion.
    
    Args:
        questions_asked: List of questions asked so far
        oracle_answers: List of oracle responses
        hl_instruction: Strategic guidance from high-level agent
        question_count: Number of questions asked
        max_questions: Maximum questions allowed
        
    Returns:
        Formatted context string presenting oracle's perspective
    """
    context_parts = [
        f"We are playing the game 20 questions. I am thinking of a word and you have asked {question_count} of {max_questions} questions."
    ]
    
    # Add the most recent oracle response if there is one
    if oracle_answers:
        last_answer = oracle_answers[-1]
        context_parts.append(f"In response to your previous question, the answer is {last_answer}.")
    
    # Add strategic advice as if coming from the oracle
    context_parts.append(f"For your next question, here is my advice: {hl_instruction}")
    
    return "\n".join(context_parts)


def format_low_level_oracle_context_optimized(
    oracle_qa_tuples: List[Tuple[str, str]],
    hl_instruction: str,
    question_count: int,
    max_questions: int = 20
) -> str:
    """
    Format the context for the low-level agent using optimized storage.
    
    This reconstructs the conversation as: LL query 1, oracle answer 1, LL query 2, ..., current HL instruction.
    Past HL instructions are NOT included to reduce context length.
    
    Args:
        oracle_qa_tuples: List of (question, answer) tuples
        hl_instruction: Current strategic guidance from high-level agent
        question_count: Number of questions asked
        max_questions: Maximum questions allowed
        
    Returns:
        Formatted context string with reduced context length
    """
    context_parts = [
        f"We are playing the game 20 questions. I am thinking of a word and you have asked {question_count} of {max_questions} questions."
    ]
    
    # Add the most recent oracle response if there is one
    if oracle_qa_tuples:
        last_question, last_answer = oracle_qa_tuples[-1]
        context_parts.append(f"In response to your previous question, the answer is {last_answer}.")
    
    # Add strategic advice as if coming from the oracle
    context_parts.append(f"For your next question, here is my advice: {hl_instruction}")
    
    return "\n".join(context_parts)


def detect_guess_instruction(hl_instruction: str) -> Optional[str]:
    """
    Detect if the high-level instruction is asking for a final guess.
    
    Args:
        hl_instruction: High-level instruction text
        
    Returns:
        Extracted word to guess, or None if not a guess instruction
    """
    instruction_lower = hl_instruction.lower()
    
    # Look for guess patterns
    guess_patterns = [
        "make your final guess",
        "time to guess", 
        "make your best guess",
        "guess now",
        "final guess",
        "i believe the word is",
        "the word is",
        "this is"
    ]
    
    for pattern in guess_patterns:
        if pattern in instruction_lower:
            # Try to extract the word from common patterns
            if "word is" in instruction_lower:
                # Extract word after "word is"
                parts = instruction_lower.split("word is")
                if len(parts) > 1:
                    word_part = parts[1].strip()
                    # Remove common punctuation and get first word
                    word_part = word_part.replace("'", "").replace('"', "").replace(".", "").replace(",", "")
                    words = word_part.split()
                    if words:
                        return words[0].capitalize()
            
            # If we detect a guess pattern but can't extract word, return indicator
            return "GUESS_DETECTED"
    
    return None


