"""Embedding model factory."""
from __future__ import annotations

from typing import Any, Dict

from rag_pipeline.utils import get_secret, logger


def _openai(cfg: Dict[str, Any]):
    from langchain_openai import OpenAIEmbeddings

    api_key = get_secret("OPENAI_API_KEY")
    kwargs: Dict[str, Any] = {"model": cfg.get("model", "text-embedding-3-small")}
    if api_key:
        kwargs["openai_api_key"] = api_key
    if cfg.get("dimensions"):
        kwargs["dimensions"] = cfg["dimensions"]
    return OpenAIEmbeddings(**kwargs)


def _huggingface(cfg: Dict[str, Any]):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=cfg.get("model", "BAAI/bge-small-en-v1.5"),
        model_kwargs={"device": cfg.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": cfg.get("normalize", True)},
    )


def _ollama(cfg: Dict[str, Any]):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=cfg.get("model", "nomic-embed-text"),
        base_url=cfg.get("base_url", "http://localhost:11434"),
    )


def _cohere(cfg: Dict[str, Any]):
    from langchain_cohere import CohereEmbeddings

    api_key = get_secret("COHERE_API_KEY")
    return CohereEmbeddings(
        model=cfg.get("model", "embed-english-light-v3.0"),
        cohere_api_key=api_key,
        user_agent=cfg.get("user_agent", "rag_pipeline"),
    )


_PROVIDERS = {
    "openai": _openai,
    "huggingface": _huggingface,
    "hf": _huggingface,
    "ollama": _ollama,
    "cohere": _cohere,
}


def build_embeddings(cfg: Dict[str, Any]):
    provider = cfg.get("provider", "huggingface").lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider: {provider}. Options: {list(_PROVIDERS)}"
        )
    logger.info("Building embeddings provider=%s model=%s", provider, cfg.get("model"))
    return _PROVIDERS[provider](cfg)
