"""
Helpers for constructing Conversation objects from raw environment state.
"""

from typing import Iterable, List, Sequence

from src.base import Conversation


def _normalize_state(state: Sequence[str]) -> List[str]:
    if isinstance(state, str):
        return [state]
    return [str(msg) for msg in state]


def conversation_from_state(state: Sequence[str]) -> Conversation:
    """Build a Conversation object from alternating user/assistant turns."""
    conv = Conversation()
    normalized = _normalize_state(state)
    if not normalized:
        return conv

    for idx, message in enumerate(normalized):
        if idx % 2 == 0:
            conv.add_user_message(message)
        else:
            conv.add_assistant_message(message)

    conv.set_original_harmful_prompt(normalized[0])
    return conv


def conversations_from_batch(states: Iterable[Sequence[str]]) -> List[Conversation]:
    """Batch helper to convert a list of conversation histories."""
    return [conversation_from_state(state) for state in states]

