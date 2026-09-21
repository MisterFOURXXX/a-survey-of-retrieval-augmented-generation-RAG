"""Modular Retrieval-Augmented Generation pipeline."""

__version__ = "0.1.0"

from rag_pipeline.config import Config
from rag_pipeline.pipeline import RAGPipeline

__all__ = ["Config", "RAGPipeline", "__version__"]
