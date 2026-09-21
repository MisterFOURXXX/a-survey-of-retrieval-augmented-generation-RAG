# %% [markdown]
# # 03 — Embeddings
# Compare HuggingFace, OpenAI, Ollama, and Cohere embedding providers.

# %%
from rag_pipeline.embeddings import build_embeddings

# %%
# HuggingFace (default, local)
hf = build_embeddings({
    "provider": "huggingface", "model": "BAAI/bge-small-en-v1.5",
    "device": "cpu", "normalize": True,
})
print("dim:", len(hf.embed_query("What is a neural network?")))

# %%
# OpenAI
# import os; os.environ["OPENAI_API_KEY"] = "..."
# openai_emb = build_embeddings({"provider": "openai",
#                                "model": "text-embedding-3-small"})
# print(len(openai_emb.embed_query("test")))

# %%
# Ollama (local; requires `ollama serve` + `ollama pull nomic-embed-text`)
# ollama_emb = build_embeddings({"provider": "ollama", "model": "nomic-embed-text"})
# print(len(ollama_emb.embed_query("test")))

# %%
# Cohere
# cohere_emb = build_embeddings({"provider": "cohere",
#                                "model": "embed-english-light-v3.0"})
# print(len(cohere_emb.embed_query("test")))
