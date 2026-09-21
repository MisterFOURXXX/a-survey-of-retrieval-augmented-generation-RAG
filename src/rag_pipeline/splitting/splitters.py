"""Text splitter factory."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from rag_pipeline.utils import logger


def _build_recursive(cfg: Dict[str, Any]):
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.get("chunk_size", 1000),
        chunk_overlap=cfg.get("chunk_overlap", 200),
        separators=cfg.get("separators", ["\n\n", "\n", " ", ""]),
        length_function=len,
    )


def _build_character(cfg: Dict[str, Any]):
    return CharacterTextSplitter(
        separator=cfg.get("separator", "\n\n"),
        chunk_size=cfg.get("chunk_size", 1000),
        chunk_overlap=cfg.get("chunk_overlap", 200),
    )


def _build_token(cfg: Dict[str, Any]):
    return TokenTextSplitter(
        chunk_size=cfg.get("chunk_size", 512),
        chunk_overlap=cfg.get("chunk_overlap", 50),
        encoding_name=cfg.get("encoding_name", "gpt2"),
    )


def _build_html(cfg: Dict[str, Any]):
    from langchain_text_splitters import HTMLHeaderTextSplitter

    headers = cfg.get(
        "headers_to_split_on",
        [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
    )
    return HTMLHeaderTextSplitter(headers_to_split_on=headers)


def _build_code(cfg: Dict[str, Any]):
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

    language = cfg.get("language", "python")
    try:
        return RecursiveCharacterTextSplitter.from_language(
            language=Language(language),
            chunk_size=cfg.get("chunk_size", 1000),
            chunk_overlap=cfg.get("chunk_overlap", 100),
        )
    except Exception:
        return _build_recursive(cfg)


def _build_semantic(cfg: Dict[str, Any]):
    from langchain_experimental.text_splitter import SemanticChunker

    embeddings = cfg.get("_embeddings")
    if embeddings is None:
        raise ValueError(
            "Semantic splitting requires '_embeddings' in the splitting config."
        )
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type=cfg.get("breakpoint_threshold_type", "percentile"),
        breakpoint_threshold_amount=cfg.get("breakpoint_threshold_amount", 95),
    )


_BUILDERS = {
    "recursive": _build_recursive,
    "character": _build_character,
    "token": _build_token,
    "html": _build_html,
    "code": _build_code,
    "semantic": _build_semantic,
}


def build_splitter(cfg: Dict[str, Any]):
    kind = cfg.get("type", "recursive")
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown splitter type: {kind}. Options: {list(_BUILDERS)}")
    return _BUILDERS[kind](cfg)


def split_documents(
    documents: List[Document],
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    cfg = cfg or {}
    splitter = build_splitter(cfg)
    chunks = splitter.split_documents(documents)
    logger.info("Split %d docs into %d chunks", len(documents), len(chunks))
    return chunks
