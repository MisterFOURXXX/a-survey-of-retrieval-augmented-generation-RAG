from __future__ import annotations
from typing import Any, Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from rag_pipeline.utils import logger

def build_retriever(vectorstore, cfg):
    st = cfg.get("search_type", "similarity")
    k = cfg.get("k", 4)
    sk = {"k": k}
    if st == "mmr":
        sk["fetch_k"] = cfg.get("fetch_k", 20)
        sk["lambda_mult"] = cfg.get("lambda_mult", 0.5)
    elif st == "similarity_score_threshold":
        sk["score_threshold"] = cfg.get("score_threshold", 0.5)
    if cfg.get("filter"): sk["filter"] = cfg["filter"]
    logger.info("Building retriever type=%s k=%s", st, k)
    return vectorstore.as_retriever(search_type=st, search_kwargs=sk)
