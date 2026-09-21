"""Retriever construction with search-type-aware kwargs."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from rag_pipeline.utils import logger


def build_retriever(
    vectorstore: VectorStore,
    cfg: Dict[str, Any],
) -> BaseRetriever:
    """Build a retriever from config.

    Supported `search_type` values:
      - similarity
      - mmr
      - similarity_score_threshold
    """
    search_type = cfg.get("search_type", "similarity")
    k = cfg.get("k", 4)

    search_kwargs: Dict[str, Any] = {"k": k}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = cfg.get("fetch_k", 20)
        search_kwargs["lambda_mult"] = cfg.get("lambda_mult", 0.5)
    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = cfg.get("score_threshold", 0.5)

    if cfg.get("filter"):
        search_kwargs["filter"] = cfg["filter"]

    logger.info("Building retriever search_type=%s k=%s", search_type, k)
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
