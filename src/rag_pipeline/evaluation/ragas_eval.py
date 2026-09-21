"""RAGAS wrapper (faithfulness + answer relevancy)."""
from __future__ import annotations
from typing import Dict, List, Optional


def run_ragas(questions: List[str], answers: List[str],
              contexts: List[List[str]],
              ground_truths: Optional[List[str]] = None) -> Dict[str, float]:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    try:
        from ragas.metrics.collections import faithfulness, answer_relevancy
    except Exception:
        from ragas.metrics import faithfulness, answer_relevancy

    data = {"question": questions, "answer": answers, "contexts": contexts}
    if ground_truths:
        data["ground_truth"] = ground_truths
    ds = Dataset.from_dict(data)
    res = ragas_evaluate(dataset=ds, metrics=[faithfulness, answer_relevancy])
    return {
        "faithfulness":     float(res["faithfulness"]),
        "answer_relevancy": float(res["answer_relevancy"]),
    }
