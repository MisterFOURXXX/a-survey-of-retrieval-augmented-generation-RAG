from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from rag_pipeline.utils import ensure_dir, logger

def _build_faiss(docs, emb, cfg):
    from langchain_community.vectorstores import FAISS
    store = FAISS.from_documents(docs, emb)
    if cfg.get("persist_dir"):
        store.save_local(str(ensure_dir(cfg["persist_dir"])))
        logger.info("Saved FAISS index to %s", cfg["persist_dir"])
    return store

def _load_faiss(emb, cfg):
    from langchain_community.vectorstores import FAISS
    return FAISS.load_local(cfg["persist_dir"], emb, allow_dangerous_deserialization=True)

def _build_chroma(docs, emb, cfg):
    from langchain_chroma import Chroma
    return Chroma.from_documents(documents=docs, embedding=emb,
        collection_name=cfg.get("collection_name", "rag_collection"),
        persist_directory=cfg.get("persist_dir"),
        collection_metadata={"hnsw:space": cfg.get("metric", "cosine")})

def _load_chroma(emb, cfg):
    from langchain_chroma import Chroma
    return Chroma(collection_name=cfg.get("collection_name", "rag_collection"),
        embedding_function=emb, persist_directory=cfg.get("persist_dir"))

_BUILDERS = {"faiss": (_build_faiss, _load_faiss), "chroma": (_build_chroma, _load_chroma)}

def build_vectorstore(docs, emb, cfg):
    kind = cfg.get("type", "faiss").lower()
    if kind not in _BUILDERS: raise ValueError(f"Unknown store: {kind}")
    return _BUILDERS[kind][0](docs, emb, cfg)

def load_vectorstore(emb, cfg):
    kind = cfg.get("type", "faiss").lower()
    if kind not in _BUILDERS: raise ValueError(f"Unknown store: {kind}")
    return _BUILDERS[kind][1](emb, cfg)
