# %% [markdown]
# # 04 — Vector Stores
# Build and visualize FAISS and Chroma indexes.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

from rag_pipeline.embeddings import build_embeddings
from rag_pipeline.ingestion import load_documents
from rag_pipeline.splitting import split_documents
from rag_pipeline.vectorstores import build_vectorstore

# %%
docs = load_documents([
    {"type": "csv", "path": "data/arxiv_data.csv", "content_columns": ["abstracts"]}
])[:500]
chunks = split_documents(docs, {
    "type": "recursive", "chunk_size": 1000, "chunk_overlap": 200,
})
emb = build_embeddings({"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"})

# %%
# FAISS
faiss_store = build_vectorstore(chunks, emb, {
    "type": "faiss",
    "persist_dir": "indexes/faiss_arxiv",
})
print("FAISS total:", faiss_store.index.ntotal)

# %%
# Chroma
chroma_store = build_vectorstore(chunks, emb, {
    "type": "chroma",
    "persist_dir": "indexes/chroma_arxiv",
    "collection_name": "arxiv",
    "metric": "cosine",
})
print("Chroma count:", chroma_store._collection.count())

# %%
# Visualize embeddings (PCA + UMAP)
texts = [c.page_content for c in chunks]
matrix = np.array(emb.embed_documents(texts))

pca = PCA(n_components=2, random_state=42).fit_transform(matrix)
umap_2d = umap.UMAP(
    n_neighbors=15, n_components=2, metric="cosine",
    min_dist=0.1, random_state=42,
).fit_transform(matrix)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(pca[:, 0], pca[:, 1], s=15, alpha=0.7)
ax1.set_title("PCA")
ax2.scatter(umap_2d[:, 0], umap_2d[:, 1], s=15, alpha=0.7)
ax2.set_title("UMAP")
plt.tight_layout()
plt.show()
