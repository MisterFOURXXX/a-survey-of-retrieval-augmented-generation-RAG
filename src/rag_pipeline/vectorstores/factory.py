"""Vector store factory supporting FAISS and Chroma."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from rag_pipeline.utils import ensure_dir, logger


# --------------------------------------------------------------------------
# FAISS
# --------------------------------------------------------------------------

def _build_faiss(
    documents: List[Document],
    embeddings: Embeddings,
    cfg: Dict[str, Any],
) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    store = FAISS.from_documents(documents, embeddings)
    if cfg.get("persist_dir"):
        path = ensure_dir(cfg["persist_dir"])
        store.save_local(str(path))
        logger.info("Saved FAISS index to %s", path)
    return store


def _load_faiss(embeddings: Embeddings, cfg: Dict[str, Any]) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    path = Path(cfg["persist_dir"])
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at {path}")
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------

def _build_chroma(
    documents: List[Document],
    embeddings: Embeddings,
    cfg: Dict[str, Any],
) -> VectorStore:
    from langchain_chroma import Chroma

    metric = cfg.get("metric", "cosine")
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=cfg.get("collection_name", "rag_collection"),
        persist_directory=cfg.get("persist_dir"),
        collection_metadata={"hnsw:space": metric},
    )


def _load_chroma(embeddings: Embeddings, cfg: Dict[str, Any]) -> VectorStore:
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=cfg.get("collection_name", "rag_collection"),
        embedding_function=embeddings,
        persist_directory=cfg.get("persist_dir"),
        collection_metadata={"hnsw:space": cfg.get("metric", "cosine")},
    )


_BUILDERS = {
    "faiss": (_build_faiss, _load_faiss),
    "chroma": (_build_chroma, _load_chroma),
}


def build_vectorstore(
    documents: List[Document],
    embeddings: Embeddings,
    cfg: Dict[str, Any],
) -> VectorStore:
    kind = cfg.get("type", "faiss").lower()
    if kind not in _BUILDERS:
        raise ValueError(
            f"Unknown vector store type: {kind}. Options: {list(_BUILDERS)}"
        )
    logger.info("Building vector store type=%s", kind)
    build_fn, _ = _BUILDERS[kind]
    return build_fn(documents, embeddings, cfg)


def load_vectorstore(
    embeddings: Embeddings,
    cfg: Dict[str, Any],
) -> VectorStore:
    kind = cfg.get("type", "faiss").lower()
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown vector store type: {kind}")
    _, load_fn = _BUILDERS[kind]
    return load_fn(embeddings, cfg)


def save_vectorstore(store: VectorStore, cfg: Dict[str, Any]) -> None:
    """Persist if the store supports it (FAISS requires an explicit call)."""
    path = cfg.get("persist_dir")
    if not path:
        return
    if hasattr(store, "save_local"):
        store.save_local(str(ensure_dir(path)))
        logger.info("Saved vector store to %s", path)
