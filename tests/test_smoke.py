from rag_pipeline.config import Config
from rag_pipeline.memory import ConversationMemory

def test_config_defaults():
    c = Config(); assert c.data == {} and c.retrieval == {}

def test_memory_window():
    m = ConversationMemory(2)
    m.add_message("user", "a"); m.add_message("assistant", "b"); m.add_message("user", "c")
    assert len(m.messages) == 2
    m.clear(); assert m.messages == []
