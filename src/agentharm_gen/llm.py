"""OpenAI backbone client for AgentHarm-Gen generators."""

import os
import re

from openai import OpenAI

BACKBONE = os.environ.get("AGENTHARM_GEN_MODEL", "gpt-4o")
MAX_TOKENS = int(os.environ.get("AGENTHARM_GEN_MAX_TOKENS", "4000"))

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def call(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = 1.0) -> str:
    """Single-turn call to the backbone model; returns raw completion text."""
    response = _get_client().chat.completions.create(
        model=BACKBONE,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


_FENCE_OPEN = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\s*\n", re.I)
_FENCE_CLOSED = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*?)```", re.S | re.I)


def _extract_fenced_code(text: str) -> str | None:
    """Extract the first ``` code block; tolerate a missing closing fence."""
    m = _FENCE_CLOSED.search(text)
    if m:
        return m.group(1).strip()

    m = _FENCE_OPEN.search(text)
    if m:
        return text[m.end() :].strip()

    return None


def extract_tag(text: str, tag: str) -> str:
    """Extract LLM output from <tag> blocks or ``` fenced code."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if m:
        return m.group(1).strip()

    m = re.search(rf"<{tag}>(.*)", text, re.S)
    if m:
        return m.group(1).strip()

    fenced = _extract_fenced_code(text)
    if fenced is not None:
        return fenced

    raise ValueError(f"No <{tag}> or ``` block found in LLM output:\n{text[:500]}")
