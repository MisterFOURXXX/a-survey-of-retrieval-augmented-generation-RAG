from __future__ import annotations
from typing import List, Tuple

class ConversationMemory:
    def __init__(self, window_size=6):
        self.window_size = window_size
        self.messages: List[Tuple[str, str]] = []
    def add_message(self, role, content):
        self.messages.append((role, content))
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]
    def get_context(self, last_n=4):
        if not self.messages: return ""
        recent = self.messages[-last_n:]
        formatted = "\n".join(f"{r.upper()}: {c}" for r, c in recent)
        return f"\nPrevious conversation:\n{formatted}\n"
    def get_stats(self):
        return {"messages": len(self.messages),
                "user_messages": sum(1 for r, _ in self.messages if r == "user")}
    def clear(self):
        self.messages = []
