# a-survey-of-retrieval-augmented-generation-RAG

A modular, config-driven implementation of a Retrieval-Augmented Generation
(RAG) pipeline built on top of LangChain. The repository refactors the original
tutorial notebook into a reusable Python package with focused experiment
notebooks (stored as jupytext percent scripts).

## Features

- **Config-driven** — every stage (loader, splitter, embeddings, vector store,
  retriever, LLM) is selected by a string in `configs/default.yaml`.
- **Lazy imports** — heavy libraries (`transformers`, `torch`, `chromadb`,
  `sentence_transformers`) are imported only where needed.
- **Extensible** — add a new loader / splitter / embedder / store by
  registering it in the corresponding factory dictionary.
- **Evaluation ready** — perplexity, BERTScore, semantic similarity, NLI, and
  RAGAS wrappers included.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configure

Copy `.env.example` to `.env` and fill in the keys you need. The pipeline
itself is configured through `configs/default.yaml`.

## Build the index

```bash
python scripts/build_index.py --config configs/default.yaml
```

Or, with `make`:

```bash
make build-index
```

## Run experiments

Notebooks are stored as jupytext percent scripts. Convert them with:

```bash
jupytext --to notebook notebooks/*.py
jupyter lab
```

Or:

```bash
make notebooks
```

## Programmatic use

```python
from rag_pipeline import RAGPipeline

pipe = RAGPipeline("configs/default.yaml")
pipe.load_index()
pipe.build_llms()

print(pipe.query("What are GANs?"))

for token in pipe.stream("Explain vision transformers"):
    print(token, end="")
```

## Package layout

- `ingestion` — loaders, cleaning, metadata enrichment
- `splitting` — recursive, HTML, JSON, token, code, semantic splitters
- `embeddings` — OpenAI / HuggingFace / Ollama / Cohere factory
- `vectorstores` — FAISS / Chroma factory
- `retrieval` — similarity, MMR, threshold, metadata-filtered retrievers
- `generation` — prompt templates + HF LLM factory
- `memory` — rolling-window conversation memory
- `evaluation` — perplexity, BERTScore, semantic, NLI, RAGAS

## License

MIT — see [LICENSE](LICENSE).
