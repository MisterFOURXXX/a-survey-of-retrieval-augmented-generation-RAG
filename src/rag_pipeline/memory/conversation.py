"""Sliding-window conversation memory."""
from __future__ import annotations

from typing import List, Tuple


class ConversationMemory:
    """Simple rolling-window memory.

    Keeps the most recent `window_size` messages and produces a formatted
    context block for injection into prompts.
    """

    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self.messages: List[Tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size :]

    def get_context(self, last_n: int = 4) -> str:
        if not self.messages:
            return ""
        recent = self.messages[-last_n:]
        formatted = "\n".join(f"{role.upper()}: {content}" for role, content in recent)
        return f"\nPrevious conversation:\n{formatted}\n"

    def get_stats(self) -> dict:
        return {
            "messages": len(self.messages),
            "user_messages": sum(1 for r, _ in self.messages if r == "user"),
        }

    def clear(self) -> None:
        self.messages = []
