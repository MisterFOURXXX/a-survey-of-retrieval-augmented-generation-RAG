"""Semantic similarity based faithfulness / answer relevance."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class SemanticEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def similarity(self, a: str, b: str) -> float:
        emb_a = self.model.encode(a, normalize_embeddings=True)
        emb_b = self.model.encode(b, normalize_embeddings=True)
        return float(np.dot(emb_a, emb_b))

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[str],
    ) -> Dict[str, float]:
        faith_scores = [
            self.similarity(ans[:200], ctx[:300])
            for ans, ctx in zip(answers, contexts)
        ]
        rel_scores = [
            self.similarity(ans[:150], q) for ans, q in zip(answers, questions)
        ]
        return {
            "faithfulness": float(np.mean(faith_scores)),
            "answer_relevance": float(np.mean(rel_scores)),
            "faithfulness_scores": faith_scores,
            "relevance_scores": rel_scores,
        }
