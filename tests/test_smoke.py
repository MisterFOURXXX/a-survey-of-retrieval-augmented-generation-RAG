"""Smoke tests: cheap paths that should always pass."""
from rag_pipeline.config import Config
from rag_pipeline.memory import ConversationMemory


def test_config_defaults():
    cfg = Config()
    assert cfg.data == {}
    assert cfg.retrieval == {}


def test_memory_window():
    m = ConversationMemory(window_size=2)
    m.add_message("user", "a")
    m.add_message("assistant", "b")
    m.add_message("user", "c")
    assert len(m.messages) == 2
    assert m.get_stats()["user_messages"] == 1
    m.clear()
    assert m.messages == []
