# %% [markdown]
# # 02 — Chunking
# Compare recursive, character, token, HTML, code, JSON, and semantic splitters.

# %%
from rag_pipeline.ingestion import load_documents
from rag_pipeline.splitting import split_documents

docs = load_documents([{
    "type": "csv", "path": "data/arxiv_data.csv", "content_columns": ["abstracts"],
}])[:100]

# %%
# Recursive (recommended default)
chunks = split_documents(docs, {
    "type": "recursive", "chunk_size": 1000, "chunk_overlap": 200,
    "separators": ["\n\n", "\n", " ", ""],
})
print(len(docs), "->", len(chunks))
print(chunks[0].page_content[:250])

# %%
# Token splitter
token_chunks = split_documents(docs, {
    "type": "token", "chunk_size": 256, "chunk_overlap": 32,
})
print(len(token_chunks), "token chunks")

# %%
# Character splitter
char_chunks = split_documents(docs, {
    "type": "character", "separator": "\n", "chunk_size": 500, "chunk_overlap": 50,
})
print(len(char_chunks), "character chunks")

# %%
# HTML header splitter
from langchain_text_splitters import HTMLHeaderTextSplitter
html = """
<h1>Foo</h1><p>Intro about foo.</p>
<h2>Bar</h2><p>Details about bar.</p>
<h3>Baz</h3><p>More details about baz.</p>
"""
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
)
for d in splitter.split_text(html):
    print(d.metadata, "|", d.page_content)

# %%
# Code splitter
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
code = "def add(a, b):\n    return a + b\n\nclass Foo:\n    pass\n" * 20
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=200, chunk_overlap=20,
)
print(len(code_splitter.split_text(code)), "code chunks")

# %%
# Semantic splitter (uses embeddings)
from rag_pipeline.embeddings import build_embeddings
emb = build_embeddings({"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"})
semantic_chunks = split_documents(docs[:5], {
    "type": "semantic", "_embeddings": emb,
})
print(len(semantic_chunks), "semantic chunks")
