"""Natural Language Inference based faithfulness."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class NLIEvaluator:
    def __init__(self, model_name: str = "roberta-large-mnli"):
        from transformers import pipeline

        self.nli = pipeline("text-classification", model=model_name)

    def _score_pair(self, premise: str, hypothesis: str) -> float:
        nli_input = f"Premise: {premise[:300]}. Hypothesis: {hypothesis[:100]}"
        result = self.nli(nli_input[:500])[0]
        label = result["label"]
        score = float(result["score"])
        if label == "ENTAILMENT":
            return score
        if label == "NEUTRAL":
            return score * 0.5
        return 0.1

    def faithfulness(self, answers: List[str], contexts: List[str]) -> float:
        scores = [self._score_pair(c, a) for a, c in zip(answers, contexts)]
        return float(np.mean(scores))

    def evaluate(
        self,
        answers: List[str],
        contexts: List[str],
    ) -> Dict[str, float]:
        return {"faithfulness": self.faithfulness(answers, contexts)}
