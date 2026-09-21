"""Semantic similarity (faithfulness / answer relevance)."""
from __future__ import annotations
from typing import Dict, List
import numpy as np


class SemanticEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def similarity(self, a: str, b: str) -> float:
        ea = self.model.encode(a, normalize_embeddings=True)
        eb = self.model.encode(b, normalize_embeddings=True)
        return float(np.dot(ea, eb))

    def evaluate(self, questions: List[str], answers: List[str],
                 contexts: List[str]) -> Dict[str, float]:
        faith = [self.similarity(a[:200], c[:300]) for a, c in zip(answers, contexts)]
        rel   = [self.similarity(a[:150], q)          for a, q in zip(answers, questions)]
        return {
            "faithfulness":          float(np.mean(faith)),
            "answer_relevance":      float(np.mean(rel)),
            "faithfulness_scores":   faith,
            "relevance_scores":      rel,
        }
