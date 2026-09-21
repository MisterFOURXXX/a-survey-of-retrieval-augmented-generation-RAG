# %% [markdown]
# # 05 — Retrieval
# Compare similarity, MMR, threshold, and metadata-filtered retrieval.

# %%
from rag_pipeline.embeddings import build_embeddings
from rag_pipeline.vectorstores import load_vectorstore
from rag_pipeline.retrieval import build_retriever

emb = build_embeddings({"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"})
store = load_vectorstore(emb, {"type": "faiss", "persist_dir": "indexes/faiss_arxiv"})

# %%
queries = [
    "Methods to improve adversarial robustness of graph neural networks",
    "Recent advances in few-shot learning for image classification",
    "Transformer-based models for long-document summarization",
]

# %%
# Standard similarity
r = build_retriever(store, {"search_type": "similarity", "k": 3})
for q in queries:
    print("Q:", q)
    for d in r.invoke(q):
        print(f"  row={d.metadata.get('row')} | {d.page_content[:80]}...")
    print()

# %%
# MMR
r_mmr = build_retriever(store, {
    "search_type": "mmr", "k": 3, "fetch_k": 20, "lambda_mult": 0.5,
})
for d in r_mmr.invoke(queries[0]):
    print(f"row={d.metadata.get('row')} | {d.page_content[:80]}...")

# %%
# Score-threshold
r_thr = build_retriever(store, {
    "search_type": "similarity_score_threshold", "k": 5, "score_threshold": 0.6,
})
for d in r_thr.invoke(queries[0]):
    print(f"row={d.metadata.get('row')} | {d.page_content[:80]}...")

# %%
# Metadata filter
r_fil = build_retriever(store, {
    "search_type": "similarity", "k": 3, "filter": {"row": 2},
})
for d in r_fil.invoke(queries[0]):
    print(f"row={d.metadata.get('row')} | {d.page_content[:80]}...")
