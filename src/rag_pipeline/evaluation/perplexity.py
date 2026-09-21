"""Perplexity evaluation via Hugging Face `evaluate`."""
from __future__ import annotations
from typing import Dict, List


def perplexity_score(predictions: List[str], model_id: str = "gpt2") -> Dict[str, float]:
    import evaluate
    metric = evaluate.load("perplexity", module_type="metric")
    result = metric.compute(predictions=predictions, model_id=model_id)
    return {"mean_perplexity": float(result["mean_perplexity"])}
