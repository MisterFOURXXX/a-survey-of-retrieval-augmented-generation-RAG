# %% [markdown]
# # 01 — Data Ingestion
# Load documents from text, CSV, PDF, JSON, and web sources, then apply
# cleaning + metadata enrichment.

# %%
from rag_pipeline.ingestion import (
    load_documents,
    clean_and_filter_documents,
    enrich_metadata,
)
from rag_pipeline.utils import setup_logging

setup_logging()

# %%
# Text
text_docs = load_documents([
    {"type": "text", "path": "data/drake_lyrics.txt"},
])
print(text_docs[0].page_content[:300])

# %%
# CSV
csv_docs = load_documents([
    {
        "type": "csv",
        "path": "data/arxiv_data.csv",
        "content_columns": ["abstracts"],
    }
])[:500]
print(csv_docs[2].page_content[:200])
print(csv_docs[2].metadata)

# %%
# PDF
pdf_docs = load_documents([
    {"type": "pdf", "path": "data/ENGINEERING/10030015.pdf"},
])
print(len(pdf_docs), "pages")
print(pdf_docs[0].page_content[:300])

# %%
# JSON with jq schema
json_docs = load_documents([
    {
        "type": "json",
        "path": "data/drake_data.json",
        "jq_schema": ".[] | .lyrics",
        "text_content": False,
    }
])
print(len(json_docs), "songs")
print(json_docs[0].page_content[:200])

# %%
# Web
web_docs = load_documents([
    {
        "type": "web",
        "urls": [
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "https://en.wikipedia.org/wiki/Java_(programming_language)",
        ],
        "parse_only_classes": ["firstHeading", "mw-parser-output"],
    }
])
print([d.metadata.get("source") for d in web_docs])

# %%
# Cleaning + metadata enrichment
cleaned = clean_and_filter_documents(csv_docs, min_length=50, max_length=10000)
enriched = enrich_metadata(cleaned)
print("Enriched sample metadata:", enriched[0].metadata)
