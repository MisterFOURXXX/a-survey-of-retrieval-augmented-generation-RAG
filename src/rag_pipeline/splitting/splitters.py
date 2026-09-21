"""Text splitter factory (recursive, character, token, HTML, code, JSON, semantic)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from rag_pipeline.utils import logger


def _build_recursive(cfg):
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.get("chunk_size", 1000),
        chunk_overlap=cfg.get("chunk_overlap", 200),
        separators=cfg.get("separators", ["\n\n", "\n", " ", ""]),
        length_function=len,
    )


def _build_character(cfg):
    return CharacterTextSplitter(
        separator=cfg.get("separator", "\n\n"),
        chunk_size=cfg.get("chunk_size", 1000),
        chunk_overlap=cfg.get("chunk_overlap", 200),
    )


def _build_token(cfg):
    return TokenTextSplitter(
        chunk_size=cfg.get("chunk_size", 512),
        chunk_overlap=cfg.get("chunk_overlap", 50),
        encoding_name=cfg.get("encoding_name", "gpt2"),
    )


def _build_html(cfg):
    from langchain_text_splitters import HTMLHeaderTextSplitter
    headers = cfg.get("headers_to_split_on", [
        ("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"),
    ])
    return HTMLHeaderTextSplitter(headers_to_split_on=headers)


def _build_code(cfg):
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


def _build_semantic(cfg):
    from langchain_experimental.text_splitter import SemanticChunker
    embeddings = cfg.get("_embeddings")
    if embeddings is None:
        raise ValueError("Semantic splitting requires '_embeddings' in the splitting config.")
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type=cfg.get("breakpoint_threshold_type", "percentile"),
        breakpoint_threshold_amount=cfg.get("breakpoint_threshold_amount", 95),
    )


def _build_json(cfg):
    from langchain_text_splitters import RecursiveJsonSplitter
    return RecursiveJsonSplitter(
        max_chunk_size=cfg.get("chunk_size", 1000),
        min_chunk_size=cfg.get("min_chunk_size", 50),
    )


_BUILDERS = {
    "recursive": _build_recursive,
    "character": _build_character,
    "token":     _build_token,
    "html":      _build_html,
    "code":      _build_code,
    "semantic":  _build_semantic,
    "json":      _build_json,
}


def build_splitter(cfg):
    kind = cfg.get("type", "recursive")
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown splitter: {kind}. Options: {list(_BUILDERS)}")
    return _BUILDERS[kind](cfg)


def split_documents(documents, cfg=None):
    cfg = cfg or {}
    chunks = build_splitter(cfg).split_documents(documents)
    logger.info("Split %d docs into %d chunks", len(documents), len(chunks))
    return chunks
