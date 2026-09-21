"""BERTScore evaluation wrapper."""
from __future__ import annotations
from typing import Dict, List


def bertscore(predictions: List[str], references: List[str],
              model_type: str = "bert-base-uncased", lang: str = "en") -> Dict[str, float]:
    import evaluate
    import numpy as np
    metric = evaluate.load("bertscore")
    result = metric.compute(
        predictions=predictions, references=references,
        lang=lang, model_type=model_type,
    )
    return {
        "precision": float(np.mean(result["precision"])),
        "recall":    float(np.mean(result["recall"])),
        "f1":        float(np.mean(result["f1"])),
    }
