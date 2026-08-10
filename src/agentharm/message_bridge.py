"""Bridge between Conversation objects and inspect_ai ChatMessage lists."""

from dataclasses import dataclass, field
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser, Model
from inspect_ai.solver import TaskState

from src.base import Conversation


def to_inspect_model_name(target_model_name: str) -> str:
    if target_model_name.startswith(("openai/", "hf/", "anthropic/", "google/")):
        return target_model_name
    openai_prefixes = ("gpt-", "o1", "o3", "o4")
    if any(target_model_name.startswith(p) for p in openai_prefixes):
        return f"openai/{target_model_name}"
    if "/" in target_model_name:
        return f"hf/{target_model_name}"
    return target_model_name


def extract_assistant_text(messages: list[ChatMessage]) -> str:
    parts = []
    for message in reversed(messages):
        if message.role == "assistant" and message.text not in (None, "", []):
            parts.append(message.text)
            break
    if not parts:
        for message in messages:
            if message.role == "assistant" and message.text not in (None, "", []):
                parts.append(message.text)
    return parts[-1] if parts else ""


def conversation_to_agent_state(conv: Conversation, inject_goal: bool) -> list[str]:
    history = []
    for user_msg, asst_msg in zip(conv.user_messages, conv.assistant_messages):
        history.append(asst_msg)
        history.append(user_msg)
    if len(conv.user_messages) > len(conv.assistant_messages):
        history.append(conv.user_messages[-1])
    if inject_goal and len(history) == 0 and conv.original_harmful_prompt:
        return [conv.original_harmful_prompt]
    return history


@dataclass
class EpisodeState:
    behavior_id: str
    task_name: str
    metadata: dict[str, Any]
    messages: list[ChatMessage] = field(default_factory=list)
    system_prompt_applied: bool = False
    task_state: TaskState | None = None
    predicted_tool: str = ""
    tools_called_before_step: list[str] = field(default_factory=list)
    step_rewards: dict[str, float] = field(default_factory=dict)
    tools_called_so_far: list[str] = field(default_factory=list)
    last_harm_score_dict: dict[str, float] = field(default_factory=dict)
    l1_option_active: bool = False
    l1_option_tool: str = ""
    l1_option_tool_index: int = -1
    l1_option_steps_remaining: int = 0
    l1_option_start_message_len: int = 0
    l1_option_state_text: str = ""

    def build_task_state(self, defender_model: Model, message_limit: int | None) -> TaskState:
        state = TaskState(
            model=defender_model,
            sample_id=self.behavior_id,
            epoch=0,
            input=self.metadata["prompt"],
            messages=list(self.messages),
            message_limit=message_limit,
            metadata=dict(self.metadata),
        )
        self.task_state = state
        return state

    def append_user_message(self, text: str) -> None:
        self.messages.append(ChatMessageUser(content=text))

    def sync_messages_from_task_state(self) -> None:
        self.messages = list(self.task_state.messages)

    def defender_response_text(self) -> str:
        return extract_assistant_text(self.messages)
