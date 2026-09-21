# %% [markdown]
# # 07 — Conversation Memory
# Maintain a rolling-window history and inject it into each prompt.

# %%
from rag_pipeline.embeddings import build_embeddings
from rag_pipeline.vectorstores import load_vectorstore
from rag_pipeline.retrieval import build_retriever
from rag_pipeline.generation import build_llms
from rag_pipeline.generation.prompts import CONVERSATION_PROMPT
from rag_pipeline.memory import ConversationMemory
from rag_pipeline.utils import format_docs

# %%
emb = build_embeddings({"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"})
store = load_vectorstore(emb, {"type": "faiss", "persist_dir": "indexes/faiss_arxiv"})
retriever = build_retriever(store, {"search_type": "similarity", "k": 2})
llm = build_llms({
    "models": [{
        "name": "Llama-3.1-8B-Instruct",
        "hf_path": "meta-llama/Llama-3.1-8B-Instruct",
        "max_new_tokens": 150,
        "load_in_4bit": True,
    }]
})["Llama-3.1-8B-Instruct"]

# %%
memory = ConversationMemory(window_size=6)

conversation = [
    "What are large language models?",
    "What are architectures used in large language models?",
    "What is LangChain?",
]

for q in conversation:
    memory.add_message("user", q)
    ctx = format_docs(retriever.invoke(q), max_chars=120)
    prompt = CONVERSATION_PROMPT.format(
        conversation_context=memory.get_context(),
        context=ctx,
        question=q,
    )
    print(f"\nUSER: {q}")
    out = ""
    for chunk in llm.stream(prompt, max_new_tokens=150):
        text = str(chunk)
        print(text, end="", flush=True)
        out += text
    memory.add_message("assistant", out[:300])
    print()

# %%
print(memory.get_stats())
for role, content in memory.messages:
    print(f"{role.upper()}: {content[:120]}...")
