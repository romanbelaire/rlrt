#!/usr/bin/env python3
"""
Generate redteam dialogue from conversation history.
Usage:
  python run.py                          # interactive
  python run.py --history '[...]' --base "harmful prompt"
"""

import argparse
import json
import os

from policy import FreeformHighLevelPolicy, LowLevelPolicy


def _history_to_state_text(history: list) -> str:
    return "\n".join(m.get("content", "") for m in history if m.get("content"))


def _history_to_state_list(history: list) -> list:
    return [m.get("content", "") for m in history if m.get("content")]


def generate(history: list, base_question: str, hl_policy, ll_policy, max_new_tokens: int = 50) -> str:
    """Generate next agent message given conversation history and base question."""
    state_text = _history_to_state_text(history)
    state_list = _history_to_state_list(history)
    instructions, _, _ = hl_policy.act([state_text], base_questions=[base_question])
    actions, _, _ = ll_policy.act(
        state=[state_list],
        base_state=[base_question],
        instructions=instructions,
        max_new_tokens=max_new_tokens,
    )
    return actions[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, help="JSON list of {role, content} messages")
    parser.add_argument("--base", type=str, default="", help="Base harmful prompt")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hl-path", type=str, default=None)
    parser.add_argument("--ll-path", type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    hl_path = args.hl_path or os.path.join(base_dir, "high_level_policy")
    ll_path = args.ll_path or os.path.join(base_dir, "policy")
    prompts_path = os.path.join(base_dir, "freeform_prompts.json")

    if not os.path.isdir(hl_path):
        raise FileNotFoundError(f"high_level_policy not found at {hl_path}")
    if not os.path.isdir(ll_path):
        raise FileNotFoundError(f"policy not found at {ll_path}")

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    hl_policy = FreeformHighLevelPolicy.load(hl_path, device=device, prompts_path=prompts_path)
    ll_policy = LowLevelPolicy.load(ll_path, device=device)

    if args.history:
        history = json.loads(args.history)
        base_question = args.base or "Continue the conversation."
        msg = generate(history, base_question, hl_policy, ll_policy, args.max_tokens)
        print(msg)
        return

    # Interactive
    print("Enter conversation history as JSON (list of {role, content}), then base question.")
    print("Example: [{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Hello\"}]")
    history_raw = input("History: ").strip()
    base_question = input("Base question: ").strip() or "Continue."
    history = json.loads(history_raw) if history_raw else []
    msg = generate(history, base_question, hl_policy, ll_policy, args.max_tokens)
    print("Generated:", msg)


if __name__ == "__main__":
    main()
