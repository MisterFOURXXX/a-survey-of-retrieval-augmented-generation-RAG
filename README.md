# A Survey of Retrieval-Augmented Generation (RAG)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, config-driven implementation of a **Retrieval-Augmented Generation (RAG)** pipeline built on top of LangChain. This repository accompanies a survey of RAG foundations, methods, and evaluation techniques. It refactors the original tutorial notebook into a reusable Python package with focused experiment notebooks.

The goal is twofold:

1. **Reproducible research** — every stage of the RAG pipeline is isolated, configurable, and independently evaluable.
2. **Pedagogical clarity** — each notebook walks through one design decision (chunking, embedding, retrieval, prompting, evaluation) with concrete examples on real datasets.

---

## Table of Contents

1. [Foundations — High-Level Design](#foundations--high-level-design)
   - [Data Ingestion](#1-data-ingestion)
   - [Chunking](#2-chunking)
   - [Embedding](#3-embedding)
   - [Vector Stores](#4-vector-stores)
   - [Retrieval](#5-retrieval)
   - [Generation & Prompt Engineering](#6-generation--prompt-engineering)
   - [Conversation Memory](#7-conversation-memory)
   - [Evaluation](#8-evaluation)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Dataset Setup](#dataset-setup)
5. [Running Experiments](#running-experiments)
6. [Notebook Reference](#notebook-reference)
7. [Configuration Reference](#configuration-reference)
8. [Results & Metrics](#results--metrics)
9. [Extending the Pipeline](#extending-the-pipeline)
10. [Citation](#citation)
11. [License](#license)

---

## Foundations — High-Level Design

A RAG system augments a large language model (LLM) with a *retrieval* step: instead of answering from parametric memory alone, the model reads relevant text fragments fetched from an external knowledge base. This grounds responses in evidence, reduces hallucination, and lets the model answer about data it was never trained on.

The pipeline decomposes into **eight stages**. Each is a first-class concern in this repository.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          INDEXING (offline)                            │
│                                                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │  Ingestion  │───>│  Chunking   │───>│  Embedding  │───>│  Vector  │ │
│  │  (loaders)  │    │ (splitters) │    │  (encoder)  │    │  Store   │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
└────────────────────────────────────────────────────────────────────────┘
                                                               │
                                                               v
┌────────────────────────────────────────────────────────────────────────┐
│                           QUERYING (online)                            │
│                                                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │    Query    │───>│  Retrieval  │───>│   Prompt    │───>│   LLM    │ │
│  │   Encoder   │    │   (top-k)   │    │  Template   │    │ (Gen)    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
│                                                                │       │
│                                                                v       │
│                                                           ┌──────────┐ │
│                                                           │  Eval    │ │
│                                                           └──────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Data Ingestion

**What it does.** Loads raw documents — PDFs, CSVs, JSON, Markdown, web pages, SQL rows, S3 objects, Git repos — and normalizes them into LangChain `Document` objects (a `page_content` string plus a `metadata` dict).

**Why it matters.** Real-world corpora are heterogeneous. A resume PDF, an arXiv abstract, a product description, and a Wikipedia article have different structures. Ingestion is where we erase those differences without losing provenance.

**How it works.**

| Loader kind | LangChain class | Preserves |
|---|---|---|
| Plain text | `TextLoader` | Lines |
| CSV / TSV | `CSVLoader` | Columns, row IDs |
| PDF | `PyPDFLoader` | Page boundaries |
| JSON | `JSONLoader` (jq) | Nested fields |
| HTML | `WebBaseLoader` | DOM structure (filterable via `SoupStrainer`) |
| SQL | `SQLDatabaseLoader` | Table rows |
| S3 / Git | `S3FileLoader`, `GitLoader` | Remote provenance |

Two production-grade practices are baked in:

- **Lazy loading** (`.lazy_load()`) — a generator yields one document at a time so a 10 GB corpus does not blow up RAM.
- **Skip-missing** — a source with `skip_missing: true` logs a warning and is skipped if its path does not exist, letting a single config work across environments.

After loading, we **clean** (strip boilerplate, collapse whitespace, remove PII) and **enrich metadata** (append `processed_at`, `content_hash`, `file_type`, `word_count`) so downstream retrievers can filter and cite.

> **Notebook:** `01_data_ingestion.ipynb`

---

### 2. Chunking

**What it does.** Splits long documents into smaller fragments (chunks) that fit the LLM's context window and the embedding model's input limit.

**Why it matters.** A 30-page PDF is useless as a single vector — the embedding model would average away all detail. Chunking balances two opposing forces:

- **Chunks too large** -> embedding blurs, retrieval loses precision, prompts waste tokens.
- **Chunks too small** -> semantic coherence is lost, retrieval misses multi-sentence reasoning.

**How it works.** The `RecursiveCharacterTextSplitter` is the default. It tries a list of separators in priority order:

```
"\n\n"  ->  "\n"  ->  " "  ->  ""     (paragraph -> line -> word -> character)
```

The splitter walks down the list until a chunk fits under `chunk_size`. `chunk_overlap` repeats the last *N* characters at the start of the next chunk so the semantic bridge between two chunks is never lost.

Other splitters available:

| Splitter | Best for |
|---|---|
| `RecursiveCharacterTextSplitter` | General prose (default) |
| `CharacterTextSplitter` | Pre-cleaned data (CSV rows, lists) |
| `TokenTextSplitter` | Precise control of the LLM context budget |
| `HTMLHeaderTextSplitter` | HTML with `<h1>`–`<h3>` structure |
| `RecursiveCharacterTextSplitter.from_language(...)` | Source code (Python, JS, C++) |
| `SemanticChunker` | Long documents where topic shifts matter |

> **Notebook:** `02_chunking.ipynb`

---

### 3. Embedding

**What it does.** Converts each chunk into a dense numeric vector (e.g. 384 floats) using a transformer encoder. Similar meanings land close together in this high-dimensional space.

**Why it matters.** Embeddings are the language of retrieval. They let us answer *"Find me passages about adversarial robustness of GNNs"* without the word `adversarial` ever appearing literally.

**How it works.**

- **Encode** each chunk once during indexing.
- **Encode** the user's query with the *same* model at query time.
- **Compare** the two vectors using a similarity metric (cosine by default).

The repository exposes four providers through one factory:

| Provider | Model | Dim | Deployment |
|---|---|---|---|
| HuggingFace | `BAAI/bge-small-en-v1.5` | 384 | Local (default) |
| OpenAI | `text-embedding-3-small` | 1536 | Cloud API |
| Ollama | `nomic-embed-text` | 768 | Local server |
| Cohere | `embed-english-light-v3.0` | 384 | Cloud API |

Swapping providers is a one-line YAML change. The same code path runs the encoder regardless of source.

> **Notebook:** `03_embeddings.ipynb`

---

### 4. Vector Stores

**What it does.** Indexes embeddings so that a nearest-neighbour query over millions of vectors returns in milliseconds.

**Why it matters.** Brute-force search over N vectors is O(N). Approximate nearest-neighbour (ANN) structures like HNSW bring that down to O(log N) at a small recall cost. Choosing the right index is a *speed/accuracy/storage* trade-off.

**How it works.**

| Store | Backing index | Persistence | Best for |
|---|---|---|---|
| **FAISS** | `IndexFlatIP` / `IndexFlatL2` / `IndexHNSWFlat` / `IndexIVFFlat` | Files on disk | Local, single-node, maximum speed |
| **Chroma** | HNSW | SQLite + parquet | Persistent, easy metadata filtering |

Distance metrics:

- **Cosine** (recommended for normalized text embeddings) — angle between vectors.
- **Inner product** — dot product; equals cosine when inputs are unit-normalised.
- **Euclidean (L2)** — raw spatial distance.

Both stores are exposed through the same `build_vectorstore(...)` / `load_vectorstore(...)` interface, so the rest of the pipeline never cares which one is behind it.

> **Notebook:** `04_vector_stores.ipynb`

---

### 5. Retrieval

**What it does.** Given a query, find the top-K most relevant chunks from the vector store.

**Why it matters.** Retrieval is where RAG either wins or fails. If the retriever surfaces irrelevant context, the LLM has no chance of producing a correct answer — garbage in, garbage out.

**How it works.** Four strategies are implemented and compared side-by-side:

| Strategy | Formula | When to use |
|---|---|---|
| **Similarity** | Top-K by cosine | Baseline, high precision |
| **MMR** | `argmax[λ·sim(d,q) − (1−λ)·max sim(d,d')]` | Diversity — avoid redundant chunks |
| **Score threshold** | Keep all with sim ≥ τ | Quality over quantity |
| **Metadata filter** | Pre-filter, then top-K | Hard constraints (date, author, source) |

The `build_retriever(store, cfg)` function maps a config dict to the right LangChain retriever — no branching logic in the notebooks.

> **Notebooks:** `05_retrieval.ipynb`, `09_retrieval_evaluation.ipynb`, `10_faiss_metric_comparison.ipynb`, `11_chroma_metric_comparison.ipynb`

---

### 6. Generation & Prompt Engineering

**What it does.** Concatenates the retrieved chunks with the user's question into a *prompt*, then feeds it to an LLM which produces the final answer.

**Why it matters.** The prompt is the contract between retrieval and generation. It tells the LLM *what* to read, *how* to read it, and *how* to cite. Bad prompts let the model drift back to its parametric memory and hallucinate.

**How it works.** The repository ships nine prompt templates:

| Template | Family | Purpose |
|---|---|---|
| `RAG_PROMPT_TEMPLATE` | Grounded QA | Answer strictly from context |
| `COT_PROMPT` | Chain-of-Thought | Step-by-step reasoning before answer |
| `ZERO_SHOT_PROMPT` | Baseline | No examples |
| `FEW_SHOT_PROMPT` | Few-shot | Two worked examples teach format |
| `REACT_PROMPT` | ReAct | Interleave Thought -> Action -> Answer |
| `TOT_PROMPT` | Tree-of-Thought | Explore multiple reasoning paths |
| `REASONING_TEMPLATE` | Structured reasoning | Explicit 9-step reasoning scaffold |
| `STRUCTURED_OUTPUT_PROMPT` | JSON schema | Q / A / summary / cited rows |
| `CONVERSATION_PROMPT` | Chat | Injects conversation memory |

Three invocation modes:

- `.invoke()` — one-shot, blocking.
- `.stream()` — token-by-token (real-time UX).
- `.batch()` — parallel processing of many prompts (with `max_concurrency` control).

> **Notebooks:** `06_generation_and_prompts.ipynb`, `12_model_invocation.ipynb`, `13_advanced_generation.ipynb`

---

### 7. Conversation Memory

**What it does.** Remembers prior turns so follow-up questions ("What did *it* change?") have context.

**Why it matters.** LLMs are stateless. Without memory, every query starts from zero — the model cannot resolve pronouns, cannot build on previous answers.

**How it works.** A rolling window keeps the last *N* messages, formatted into a compact context block:

```
Previous conversation:
USER: What are LLMs?
ASSISTANT: Large language models are...
USER: What architectures do they use?
```

The window caps at `window_size` so long chats don't blow past the context window. Summary-based and token-based memory can be dropped in later by replacing `ConversationMemory`.

> **Notebook:** `07_memory.ipynb`

---

### 8. Evaluation

**What it does.** Measures whether the retrieved context was relevant and whether the generated answer was faithful.

**Why it matters.** RAG can fail in three distinct ways, and a single "pass/fail" score cannot tell you which:

1. **Retrieval failure** — context is irrelevant.
2. **Generation failure** — context is fine, but the model hallucinated.
3. **Relevance failure** — model answered the wrong question.

The **RAG Triad** isolates all three.

**How it works.**

| Metric | Axis | What it measures |
|---|---|---|
| **Context Relevance** | Query -> Context | Did retrieval find the right chunks? |
| **Faithfulness** | Context -> Answer | Is every claim supported by the context? |
| **Answer Relevance** | Query -> Answer | Did the answer address the question? |
| **Hit Rate@K** | Ranking | Is ground truth in the top-K? |
| **MRR** | Ranking | How high did ground truth rank? |
| **nDCG@K** | Ranking | Position-weighted relevance |
| **Perplexity** | Fluency | Does the text look natural? |
| **BERTScore** | Semantic | How close is the answer to a reference? |
| **NLI** | Entailment | Does context logically entail the answer? |

All metrics are implemented in `rag_pipeline.evaluation` — no external dependencies beyond `evaluate`, `bert-score`, and `sentence-transformers`.

> **Notebooks:** `08_evaluation.ipynb`, `09_retrieval_evaluation.ipynb`

---

## Repository Structure

```
a-survey-of-retrieval-augmented-generation-RAG/
├── README.md                        ← this file
├── LICENSE                          ← MIT
├── Makefile                         ← build / test / notebook targets
├── pyproject.toml                   ← pip install -e .
├── requirements.txt                 ← runtime dependencies
├── requirements-dev.txt             ← dev tooling
├── .env.example                     ← copy to .env and fill keys
├── .gitignore
├── .pre-commit-config.yaml
│
├── configs/
│   └── default.yaml                 ← every stage's parameters
│
├── data/                            ← datasets (gitignored)
│   ├── arxiv_data.csv               ← arXiv abstracts (summaries column)
│   ├── drake_lyrics.txt             ← Drake lyrics (plain text)
│   ├── drake_data.json              ← Drake lyrics (structured)
│   └── ENGINEERING/10030015.pdf     ← resume dataset sample
│
├── indexes/                         ← persisted vector stores (gitignored)
│   ├── faiss_index/
│   └── chroma_arxiv/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_chunking.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_vector_stores.ipynb
│   ├── 05_retrieval.ipynb
│   ├── 06_generation_and_prompts.ipynb
│   ├── 07_memory.ipynb
│   ├── 08_evaluation.ipynb
│   ├── 09_retrieval_evaluation.ipynb
│   ├── 10_faiss_metric_comparison.ipynb
│   ├── 11_chroma_metric_comparison.ipynb
│   ├── 12_model_invocation.ipynb
│   └── 13_advanced_generation.ipynb
│
├── scripts/
│   └── build_index.py               ← CLI: build the vector index
│
├── src/
│   └── rag_pipeline/
│       ├── __init__.py
│       ├── config.py                ← Pydantic Config
│       ├── utils.py                 ← logging, secrets, notebook helpers
│       ├── pipeline.py              ← RAGPipeline facade
│       ├── ingestion/               ← loaders, cleaners, metadata
│       ├── splitting/               ← splitter factory
│       ├── embeddings/              ← provider factory
│       ├── vectorstores/            ← FAISS + Chroma factory
│       ├── retrieval/               ← retriever builder
│       ├── generation/              ← prompts + HF LLM factory
│       ├── memory/                  ← conversation buffer
│       └── evaluation/              ← perplexity, BERTScore, NLI, RAGAS
│
└── tests/
    └── test_smoke.py
```

---

## Installation

### Prerequisites

- Python **3.10+**
- ~30 GB free disk (HuggingFace model weights)
- GPU with **≥ 8 GB VRAM** for Llama-3.1-8B inference (4-bit quantised). CPU works but is very slow.
- A **HuggingFace token** with `read` scope. Accept the Llama license at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### Steps

```bash
# 1. Clone
git clone https://github.com/MisterFOURXXX/a-survey-of-retrieval-augmented-generation-RAG
cd a-survey-of-retrieval-augmented-generation-RAG

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

# 3. Install runtime + editable package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. (Optional) install dev tooling
pip install -r requirements-dev.txt
```

Verify:

```bash
python -c "import rag_pipeline; print(rag_pipeline.__version__)"
# -> 0.1.0
```

> **Colab users** — `make` is not preinstalled. Run `!apt-get -qq install -y make` once per session.

### Environment file

```bash
cp .env.example .env
```

Edit `.env`:

```bash
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
# OPENAI_API_KEY=sk-...      # only if using OpenAI embeddings / RAGAS
# COHERE_API_KEY=...         # only if using Cohere embeddings
HF_HOME=./.cache/huggingface
TOKENIZERS_PARALLELISM=false
```

Then log in to HuggingFace once so gated models can be downloaded:

```bash
huggingface-cli login
```

---

## Dataset Setup

This repository uses three Kaggle datasets. They are downloaded into `data/` at the repo root. The download commands below work in Colab and locally.

### Step 1 — Configure Kaggle credentials

Download `kaggle.json` from **<https://www.kaggle.com/settings/account>** -> *Create New API Token*. Then:

```bash
pip install kaggle
mkdir -p ~/.kaggle
mv /content/kaggle.json ~/.kaggle/    # or your local path
chmod 600 ~/.kaggle/kaggle.json
```

Verify:

```bash
python -c "import kaggle; kaggle.api.authenticate(); print('OK')"
```

### Step 2 — Download the datasets

```python
import kaggle
kaggle.api.authenticate()

DATA = "/content/a-survey-of-retrieval-augmented-generation-RAG/data"

# Resume dataset (PDFs for the PDF loader demo)
kaggle.api.dataset_download_files(
    "snehaanbhawal/resume-dataset", path=DATA, unzip=True,
)

# Drake lyrics (text + JSON loaders)
kaggle.api.dataset_download_files(
    "juicobowley/drake-lyrics", path=DATA, unzip=True,
)

# arXiv paper abstracts (main corpus — CSV loader)
kaggle.api.dataset_download_files(
    "spsayakpaul/arxiv-paper-abstracts", path=DATA, unzip=True,
)
```

### Step 3 — Verify `data/`

After downloading, `data/` should look like:

```
data/
├── arxiv_data.csv                 ← main corpus (51,774 rows)
├── drake_lyrics.txt               ← Drake lyrics (plain text)
├── drake_data.json                ← Drake lyrics (structured)
├── Tags.csv                       ← resume dataset metadata
├── Questions.csv
├── Answers.csv
└── ENGINEERING/
    └── 10030015.pdf               ← resume sample
```

> **Column note.** The arXiv CSV used here has columns `titles`, `summaries`, `terms`. The default config sets `content_columns: [summaries]`. If you swap in the original `spsayakpaul` release with an `abstracts` column, change the config accordingly.

---

## Running Experiments

There are two ways to run the pipeline: as **notebooks** (for exploration) or as a **script** (for reproducibility).

### Option A — Notebooks (recommended for first-timers)

```bash
# 1. Convert jupytext scripts to real .ipynb
pip install -q jupytext
jupytext --to notebook notebooks/*.py

# 2. Launch Jupyter
jupyter lab
```

Inside Jupyter, run the notebooks **in order**:

| Order | Notebook | Purpose | Typical runtime |
|---|---|---|---|
| 1 | `01_data_ingestion` | Load + clean + enrich | ~10 s |
| 2 | `02_chunking` | Compare splitters | ~30 s |
| 3 | `04_vector_stores` | Build FAISS + Chroma indexes | 3–5 min (CPU) |
| 4 | `05_retrieval` | Compare retrievers | ~10 s |
| 5 | `06_generation_and_prompts` | Compare prompts | 2–3 min (GPU) |
| 6 | `07_memory` | Multi-turn conversation | ~1 min |
| 7 | `09_retrieval_evaluation` | Hit@K, MRR, nDCG | ~30 s |
| 8 | `10_faiss_metric_comparison` | Cosine vs IP vs L2 (FAISS) | ~2 min |
| 9 | `11_chroma_metric_comparison` | Cosine vs IP vs L2 (Chroma) | ~3 min |
| 10 | `12_model_invocation` | `.invoke()` / `.stream()` / `.batch()` | ~2 min |
| 11 | `13_advanced_generation` | Structured + reasoning | ~3 min |
| 12 | `08_evaluation` | Perplexity, BERTScore, NLI, RAGAS | ~5 min |

> `03_embeddings` can be run any time — it only compares embedding providers.

### Option B — Scripts (recommended for reproducibility)

**Build the index** from the CLI:

```bash
python scripts/build_index.py --config configs/default.yaml
# or
make build-index
```

Expected output:

```
INFO | rag_pipeline | Loaded 51774 documents from 1 source(s)
INFO | rag_pipeline | Sampled 5177 / 51774 documents (fraction=0.1, seed=42)
INFO | rag_pipeline | Cleaned 5177 -> 5177 documents
INFO | rag_pipeline | Split 5177 docs into ~8000 chunks
INFO | rag_pipeline | Saved FAISS index to indexes/faiss_index
Index built and persisted.
```

**Load and query** from Python:

```python
from rag_pipeline import RAGPipeline

pipe = RAGPipeline("configs/default.yaml")
pipe.load_index()
pipe.build_llms()

# One-shot
print(pipe.query("What are GANs?"))

# Streaming
for token in pipe.stream("Explain vision transformers"):
    print(token, end="")

# Multi-turn (memory tracked automatically)
pipe.query("What are LLMs?", track_memory=True)
pipe.query("What architectures do they use?", track_memory=True)
print(pipe.memory.get_stats())
```

### Colab quickstart

Paste the following into a single Colab cell:

```python
# Clone + install
!git clone https://github.com/MisterFOURXXX/a-survey-of-retrieval-augmented-generation-RAG
%cd a-survey-of-retrieval-augmented-generation-RAG
!pip install -q -r requirements.txt && pip install -q -e .

# Make + Kaggle
!apt-get -qq install -y make
!mkdir -p ~/.kaggle && mv /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Datasets
import kaggle
kaggle.api.authenticate()
DATA = "/content/a-survey-of-retrieval-augmented-generation-RAG/data"
kaggle.api.dataset_download_files("snehaanbhawal/resume-dataset",       path=DATA, unzip=True)
kaggle.api.dataset_download_files("juicobowley/drake-lyrics",           path=DATA, unzip=True)
kaggle.api.dataset_download_files("spsayakpaul/arxiv-paper-abstracts",  path=DATA, unzip=True)

# Environment
!cp .env.example .env
# Then edit .env and paste your HF token, or:
# import os; os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

# Build the index
!make build-index
```

---

## Notebook Reference

| # | Notebook | Key questions it answers |
|---|---|---|
| 01 | Data Ingestion | How do I load text, CSV, PDF, JSON, and web sources into a uniform representation? |
| 02 | Chunking | Which splitter should I use? How does `chunk_size` affect downstream recall? |
| 03 | Embeddings | How do HuggingFace / OpenAI / Ollama / Cohere compare? |
| 04 | Vector Stores | How do FAISS and Chroma differ? When should I pick each? |
| 05 | Retrieval | Similarity vs MMR vs threshold vs metadata filter — trade-offs? |
| 06 | Generation & Prompts | How does the prompt template change the model's answer? |
| 07 | Memory | How do I make multi-turn conversations coherent? |
| 08 | Evaluation | Is my answer faithful? Is it relevant? |
| 09 | Retrieval Evaluation | Hit@K, MRR, nDCG on a synthetic labelled test set |
| 10 | FAISS Metric Comparison | cosine / IP / L2 — same question, three answers |
| 11 | Chroma Metric Comparison | Same, on Chroma |
| 12 | Model Invocation | `.invoke()` vs `.stream()` vs `.batch()` |
| 13 | Advanced Generation | Structured output + step-by-step reasoning |

---

## Configuration Reference

Every parameter that affects the pipeline lives in **`configs/default.yaml`**:

```yaml
data:
  sample_fraction: 0.1                    # keep 10% of the corpus
  sample_size: null                       # or a fixed count
  sample_seed: 42                         # reproducible sampling
  sources:
    - type: csv
      path: data/arxiv_data.csv
      content_columns: [summaries]
      skip_missing: true

paths:
  faiss_index: indexes/faiss_index
  chroma_index: indexes/chroma_arxiv

cleaning:
  min_length: 50
  max_length: 10000

splitting:
  type: recursive
  chunk_size: 1000
  chunk_overlap: 200

embeddings:
  provider: huggingface                   # openai | huggingface | ollama | cohere
  model: BAAI/bge-small-en-v1.5

vectorstore:
  type: faiss                             # faiss | chroma
  metric: cosine

retrieval:
  search_type: similarity                 # similarity | mmr | similarity_score_threshold
  k: 3

llm:
  models:
    - name: Llama-3.1-8B-Instruct
      hf_path: meta-llama/Llama-3.1-8B-Instruct
      load_in_4bit: true
```

Secrets (API keys, tokens) are **never** in the YAML — they come from `.env`:

```bash
HUGGINGFACEHUB_API_TOKEN=hf_...
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```
---

## Citation

If you use this repository in research, please cite:

```bibtex
@misc{ragsurvey2025,
  title  = {A Survey of Retrieval-Augmented Generation: Modular Pipeline and Experiments},
  author = {MisterFOURXXX},
  year   = {2025},
  url    = {https://github.com/MisterFOURXXX/a-survey-of-retrieval-augmented-generation-RAG}
}
```

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
- Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- LangChain documentation — <https://python.langchain.com/>
- Chroma documentation — <https://docs.trychroma.com/>
- FAISS documentation — <https://faiss.ai/>

---

## License

MIT — see [LICENSE](LICENSE).
