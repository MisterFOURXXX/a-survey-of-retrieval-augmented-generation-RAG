from __future__ import annotations
from typing import Any, Dict
from rag_pipeline.utils import get_secret, logger

def _openai(cfg):
    from langchain_openai import OpenAIEmbeddings
    kw = {"model": cfg.get("model", "text-embedding-3-small")}
    key = get_secret("OPENAI_API_KEY")
    if key: kw["openai_api_key"] = key
    return OpenAIEmbeddings(**kw)

def _huggingface(cfg):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=cfg.get("model", "BAAI/bge-small-en-v1.5"),
        model_kwargs={"device": cfg.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": cfg.get("normalize", True)})

def _ollama(cfg):
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=cfg.get("model", "nomic-embed-text"),
                            base_url=cfg.get("base_url", "http://localhost:11434"))

def _cohere(cfg):
    from langchain_cohere import CohereEmbeddings
    return CohereEmbeddings(model=cfg.get("model", "embed-english-light-v3.0"),
                            cohere_api_key=get_secret("COHERE_API_KEY"))

_PROVIDERS = {"openai": _openai, "huggingface": _huggingface, "hf": _huggingface,
              "ollama": _ollama, "cohere": _cohere}

def build_embeddings(cfg):
    p = cfg.get("provider", "huggingface").lower()
    if p not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {p}")
    logger.info("Building embeddings provider=%s model=%s", p, cfg.get("model"))
    return _PROVIDERS[p](cfg)
