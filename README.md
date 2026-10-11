# A Survey of Retrieval-Augmented Generation (RAG)

> A modular, config-driven implementation of a Retrieval-Augmented Generation (RAG) pipeline built on LangChain. It covers eight stages: data ingestion, chunking, embedding, vector stores, retrieval, generation & prompt engineering, conversation memory, and evaluation — with reproducible experiment notebooks on real datasets.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/langchain-%3E%3D0.2-1c3c3c.svg)](https://python.langchain.com/)
[![HuggingFace](https://img.shields.io/badge/huggingface-%3E%3D4.57-yellow.svg)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/faiss-%3E%3D1.7-4b8bbe.svg)](https://faiss.ai/)
[![Chroma](https://img.shields.io/badge/chroma-%3E%3D0.4-ff6f61.svg)](https://docs.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

This repository is a **reproducible empirical survey** of Retrieval-Augmented Generation (RAG) pipelines. It provides a modular, configuration-driven implementation built on LangChain that decomposes RAG into eight first-class stages: data ingestion, chunking, embedding, vector stores, retrieval, generation and prompt engineering, conversation memory, and evaluation. Each stage is implemented behind a uniform factory interface, driven entirely by a YAML configuration file, and exercised through thirteen reproducible experiment notebooks on real datasets (arXiv abstracts, Drake lyrics, and a resume corpus). The work is positioned as a **systems-level empirical survey of the RAG design space**: rather than proposing a new retrieval algorithm, it characterises the design choices, trade-offs, and failure modes at every stage of the pipeline under a fixed evaluation protocol. The repository contributes (i) a unified implementation across four loader families, six splitter strategies, four embedding providers, two vector stores, four retrieval strategies, nine prompt templates, and eleven evaluation metrics; (ii) a side-by-side comparison of distance metrics (cosine, inner product, L2) on both FAISS and Chroma; (iii) a structured evaluation framework based on the RAG Triad (Context Relevance, Faithfulness, Answer Relevance) plus ranking metrics (Hit@K, MRR, nDCG) and semantic metrics (BERTScore, NLI, perplexity); and (iv) a fully reproducible engineering pipeline with secrets separated from configuration, a `RAGPipeline` facade, and a `Makefile`-driven build path. It is designed to be the **retrieval-augmented generation companion** to the broader LLM engineering survey series.

**Keywords:** Retrieval-Augmented Generation, RAG, LangChain, vector databases, FAISS, Chroma, embeddings, prompt engineering, RAG Triad, evaluation, empirical survey, reproducible ML.

---

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction and Research Context](#1-introduction-and-research-context)
  - [1.1 Contributions](#11-contributions)
- [2. Foundations — High-Level Design](#2-foundations--high-level-design)
  - [2.1 Data Ingestion](#21-data-ingestion)
  - [2.2 Chunking](#22-chunking)
  - [2.3 Embedding](#23-embedding)
  - [2.4 Vector Stores](#24-vector-stores)
  - [2.5 Retrieval](#25-retrieval)
  - [2.6 Generation & Prompt Engineering](#26-generation--prompt-engineering)
  - [2.7 Conversation Memory](#27-conversation-memory)
  - [2.8 Evaluation](#28-evaluation)
- [3. Approach](#3-approach)
  - [3.1 Survey Methodology](#31-survey-methodology)
  - [3.2 Fixed Evaluation Protocol](#32-fixed-evaluation-protocol)
  - [3.3 Repository Structure](#33-repository-structure)
- [4. Evaluation](#4-evaluation)
  - [4.1 Evaluation Metrics](#41-evaluation-metrics)
  - [4.2 Notebook Reference](#42-notebook-reference)
  - [4.3 Results & Metrics](#43-results--metrics)
- [5. Installation and Dataset Setup](#5-installation-and-dataset-setup)
  - [5.1 Prerequisites](#51-prerequisites)
  - [5.2 Installation Steps](#52-installation-steps)
  - [5.3 Environment File](#53-environment-file)
  - [5.4 Dataset Setup](#54-dataset-setup)
- [6. Running Experiments](#6-running-experiments)
  - [6.1 Option A — Notebooks](#61-option-a--notebooks)
  - [6.2 Option B — Scripts](#62-option-b--scripts)
  - [6.3 Colab Quickstart](#63-colab-quickstart)
- [7. Configuration Reference](#7-configuration-reference)
- [8. Extending the Pipeline](#8-extending-the-pipeline)
- [9. Conclusion](#9-conclusion)
- [10. References](#10-references)
- [11. License](#11-license)
- [12. About](#12-about)

---

## 1. Introduction and Research Context

A RAG system augments a large language model (LLM) with a retrieval step: instead of answering from parametric memory alone, the model reads relevant text fragments fetched from an external knowledge base. This grounds responses in evidence, reduces hallucination, and lets the model answer about data it was never trained on.

The pipeline decomposes into eight stages. Each is a first-class concern in this repository.

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

This repository is a **survey as an artefact**: it implements every stage, exposes every design choice as a configuration knob, and provides reproducible experiment notebooks so that the reader can measure the trade-offs directly rather than reading about them abstractly.

### 1.1 Contributions

1. A unified implementation across four loader families, six splitter strategies, four embedding providers, two vector stores, four retrieval strategies, nine prompt templates, and eleven evaluation metrics — all behind uniform factory interfaces.
2. A side-by-side comparison of distance metrics (cosine, inner product, L2) on both FAISS and Chroma.
3. A structured evaluation framework based on the **RAG Triad** (Context Relevance, Faithfulness, Answer Relevance) plus ranking metrics (Hit@K, MRR, nDCG) and semantic metrics (BERTScore, NLI, perplexity).
4. A fully reproducible engineering pipeline: YAML-driven orchestration, secrets separated from configuration, a `RAGPipeline` facade, and a `Makefile`-driven build path.
5. Thirteen experiment notebooks on real datasets (arXiv abstracts, Drake lyrics, resume corpus), each answering a specific design question.
6. An explicit attribution, licensing, and dual-use statement for the datasets, models, and code.

---

## 2. Foundations — High-Level Design

This chapter documents the eight stages of the RAG pipeline. Each is treated as a first-class concern with its own design space, trade-offs, and implementation entry point.

### 2.1 Data Ingestion

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

**Notebook:** `01_data_ingestion.ipynb`

### 2.2 Chunking

**What it does.** Splits long documents into smaller fragments (chunks) that fit the LLM's context window and the embedding model's input limit.

**Why it matters.** A 30-page PDF is useless as a single vector — the embedding model would average away all detail. Chunking balances two opposing forces:

- **Chunks too large** → embedding blurs, retrieval loses precision, prompts waste tokens.
- **Chunks too small** → semantic coherence is lost, retrieval misses multi-sentence reasoning.

**How it works.** The `RecursiveCharacterTextSplitter` is the default. It tries a list of separators in priority order:

```
"\n\n"  ->  "\n"  ->  " "  ->  ""     (paragraph -> line -> word -> character)
```

The splitter walks down the list until a chunk fits under `chunk_size`. `chunk_overlap` repeats the last N characters at the start of the next chunk so the semantic bridge between two chunks is never lost.

Other splitters available:

| Splitter | Best for |
|---|---|
| `RecursiveCharacterTextSplitter` | General prose (default) |
| `CharacterTextSplitter` | Pre-cleaned data (CSV rows, lists) |
| `TokenTextSplitter` | Precise control of the LLM context budget |
| `HTMLHeaderTextSplitter` | HTML with `<h1>`–`<h3>` structure |
| `RecursiveCharacterTextSplitter.from_language(...)` | Source code (Python, JS, C++) |
| `SemanticChunker` | Long documents where topic shifts matter |

**Notebook:** `02_chunking.ipynb`

### 2.3 Embedding

**What it does.** Converts each chunk into a dense numeric vector (e.g. 384 floats) using a transformer encoder. Similar meanings land close together in this high-dimensional space.

**Why it matters.** Embeddings are the language of retrieval. They let us answer "Find me passages about adversarial robustness of GNNs" without the word *adversarial* ever appearing literally.

**How it works.**

1. Encode each chunk once during indexing.
2. Encode the user's query with the same model at query time.
3. Compare the two vectors using a similarity metric (cosine by default).

The repository exposes four providers through one factory:

| Provider | Model | Dim | Deployment |
|---|---|---|---|
| HuggingFace | `BAAI/bge-small-en-v1.5` | 384 | Local (default) |
| OpenAI | `text-embedding-3-small` | 1536 | Cloud API |
| Ollama | `nomic-embed-text` | 768 | Local server |
| Cohere | `embed-english-light-v3.0` | 384 | Cloud API |

Swapping providers is a one-line YAML change. The same code path runs the encoder regardless of source.

**Notebook:** `03_embeddings.ipynb`

### 2.4 Vector Stores

**What it does.** Indexes embeddings so that a nearest-neighbour query over millions of vectors returns in milliseconds.

**Why it matters.** Brute-force search over N vectors is `O(N)`. Approximate nearest-neighbour (ANN) structures like HNSW bring that down to `O(log N)` at a small recall cost. Choosing the right index is a speed/accuracy/storage trade-off.

**How it works.**

| Store | Backing index | Persistence | Best for |
|---|---|---|---|
| FAISS | `IndexFlatIP` / `IndexFlatL2` / `IndexHNSWFlat` / `IndexIVFFlat` | Files on disk | Local, single-node, maximum speed |
| Chroma | HNSW | SQLite + parquet | Persistent, easy metadata filtering |

**Distance metrics:**

- **Cosine** (recommended for normalized text embeddings) — angle between vectors.
- **Inner product** — dot product; equals cosine when inputs are unit-normalised.
- **Euclidean (L2)** — raw spatial distance.

Both stores are exposed through the same `build_vectorstore(...)` / `load_vectorstore(...)` interface, so the rest of the pipeline never cares which one is behind it.

**Notebook:** `04_vector_stores.ipynb`

### 2.5 Retrieval

**What it does.** Given a query, find the top-K most relevant chunks from the vector store.

**Why it matters.** Retrieval is where RAG either wins or fails. If the retriever surfaces irrelevant context, the LLM has no chance of producing a correct answer — garbage in, garbage out.

**How it works.** Four strategies are implemented and compared side-by-side:

| Strategy | Formula | When to use |
|---|---|---|
| Similarity | Top-K by cosine | Baseline, high precision |
| MMR | `argmax[λ·sim(d,q) − (1−λ)·max sim(d,d')]` | Diversity — avoid redundant chunks |
| Score threshold | Keep all with `sim ≥ τ` | Quality over quantity |
| Metadata filter | Pre-filter, then top-K | Hard constraints (date, author, source) |

The `build_retriever(store, cfg)` function maps a config dict to the right LangChain retriever — no branching logic in the notebooks.

**Notebooks:** `05_retrieval.ipynb`, `09_retrieval_evaluation.ipynb`, `10_faiss_metric_comparison.ipynb`, `11_chroma_metric_comparison.ipynb`

### 2.6 Generation & Prompt Engineering

**What it does.** Concatenates the retrieved chunks with the user's question into a prompt, then feeds it to an LLM which produces the final answer.

**Why it matters.** The prompt is the **contract between retrieval and generation**. It tells the LLM what to read, how to read it, and how to cite. Bad prompts let the model drift back to its parametric memory and hallucinate.

**How it works.** The repository ships nine prompt templates:

| Template | Family | Purpose |
|---|---|---|
| `RAG_PROMPT_TEMPLATE` | Grounded QA | Answer strictly from context |
| `COT_PROMPT` | Chain-of-Thought | Step-by-step reasoning before answer |
| `ZERO_SHOT_PROMPT` | Baseline | No examples |
| `FEW_SHOT_PROMPT` | Few-shot | Two worked examples teach format |
| `REACT_PROMPT` | ReAct | Interleave Thought → Action → Answer |
| `TOT_PROMPT` | Tree-of-Thought | Explore multiple reasoning paths |
| `REASONING_TEMPLATE` | Structured reasoning | Explicit 9-step reasoning scaffold |
| `STRUCTURED_OUTPUT_PROMPT` | JSON schema | Q / A / summary / cited rows |
| `CONVERSATION_PROMPT` | Chat | Injects conversation memory |

**Three invocation modes:**

- `.invoke()` — one-shot, blocking.
- `.stream()` — token-by-token (real-time UX).
- `.batch()` — parallel processing of many prompts (with `max_concurrency` control).

**Notebooks:** `06_generation_and_prompts.ipynb`, `12_model_invocation.ipynb`, `13_advanced_generation.ipynb`

### 2.7 Conversation Memory

**What it does.** Remembers prior turns so follow-up questions ("What did it change?") have context.

**Why it matters.** LLMs are stateless. Without memory, every query starts from zero — the model cannot resolve pronouns, cannot build on previous answers.

**How it works.** A rolling window keeps the last N messages, formatted into a compact context block:

```
Previous conversation:
USER: What are LLMs?
ASSISTANT: Large language models are...
USER: What architectures do they use?
```

The window caps at `window_size` so long chats don't blow past the context window. Summary-based and token-based memory can be dropped in later by replacing `ConversationMemory`.

**Notebook:** `07_memory.ipynb`

### 2.8 Evaluation

**What it does.** Measures whether the retrieved context was relevant and whether the generated answer was faithful.

**Why it matters.** RAG can fail in three distinct ways, and a single "pass/fail" score cannot tell you which:

1. **Retrieval failure** — context is irrelevant.
2. **Generation failure** — context is fine, but the model hallucinated.
3. **Relevance failure** — model answered the wrong question.

The **RAG Triad** isolates all three.

**How it works.**

| Metric | Axis | What it measures |
|---|---|---|
| Context Relevance | Query → Context | Did retrieval find the right chunks? |
| Faithfulness | Context → Answer | Is every claim supported by the context? |
| Answer Relevance | Query → Answer | Did the answer address the question? |
| Hit Rate@K | Ranking | Is ground truth in the top-K? |
| MRR | Ranking | How high did ground truth rank? |
| nDCG@K | Ranking | Position-weighted relevance |
| Perplexity | Fluency | Does the text look natural? |
| BERTScore | Semantic | How close is the answer to a reference? |
| NLI | Entailment | Does context logically entail the answer? |

All metrics are implemented in `rag_pipeline.evaluation` — no external dependencies beyond `evaluate`, `bert-score`, and `sentence-transformers`.

**Notebooks:** `08_evaluation.ipynb`, `09_retrieval_evaluation.ipynb`

---

## 3. Approach

This chapter documents the empirical survey methodology that underlies the repository. Because this work is a **survey** rather than a hypothesis-driven experiment, the methodological emphasis is on **controlled comparison, fixed protocols, and reproducible orchestration** rather than on statistical inference over treatment effects.

### 3.1 Survey Methodology

The repository is structured around a single principle: **every design choice in the RAG pipeline is exposed as a configuration knob**, and every configuration is exercised through a reproducible experiment notebook on real data. This is the standard methodology of controlled empirical comparison, applied to the RAG design space.

Each stage of the pipeline is implemented behind a small factory interface:

- **Loader factory** — a `type` field in the config selects the right LangChain loader.
- **Splitter factory** — a `type` field selects the right splitter strategy.
- **Embedding factory** — a `provider` field selects HuggingFace, OpenAI, Ollama, or Cohere.
- **Vector store factory** — a `type` field selects FAISS or Chroma.
- **Retriever factory** — a `search_type` field selects similarity, MMR, or threshold search.
- **Prompt registry** — nine named templates, each a first-class object.
- **Memory class** — a rolling-window conversation buffer.

This design has three consequences:

1. **No stage-specific implementation drift.** Because every stage delegates to the same upstream LangChain primitive, differences in observed behaviour are attributable to the design choices themselves, not to divergent re-implementations.
2. **YAML is the single source of truth.** Every hyperparameter — chunk size, overlap, embedding provider, top-K, metric, prompt template — lives in a version-controlled YAML file. Secrets live in `.env` and are never written to the config.
3. **Notebooks answer one question each.** Every notebook in the series isolates a single design question (e.g., "Which splitter should I use?", "How does the prompt template change the answer?"), so the reader can measure the trade-off directly rather than reading about it abstractly.

### 3.2 Fixed Evaluation Protocol

To make the comparison meaningful, the survey fixes the following across all experiments:

- **Datasets:** three real corpora — arXiv abstracts (main corpus, ~51,774 rows), Drake lyrics (text + JSON loaders), and a resume PDF corpus. The default config samples 10% of the arXiv corpus with `sample_seed: 42` for reproducibility.
- **Embedding model:** `BAAI/bge-small-en-v1.5` (384-dim) for all baseline experiments; other providers are compared in `03_embeddings.ipynb`.
- **Vector store:** FAISS for baseline experiments; Chroma is compared in `11_chroma_metric_comparison.ipynb`.
- **Retriever:** similarity search with `k=3` for baseline experiments; MMR and threshold search are compared in `05_retrieval.ipynb`.
- **LLM:** `meta-llama/Llama-3.1-8B-Instruct` in 4-bit quantization, loaded through the Hugging Face pipeline.
- **Evaluation:** the RAG Triad plus ranking and semantic metrics, computed on a synthetic labelled test set for retrieval evaluation and on held-out reference answers for generation evaluation.

Because the datasets, embedding model, vector store, and LLM are identical across baseline experiments, the observed differences are attributable to the design choices under study rather than to confounds in the pipeline.

### 3.3 Repository Structure

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

## 4. Evaluation

### 4.1 Evaluation Metrics

Evaluation in this repository happens at two distinct stages of the pipeline, and each stage tracks a different family of metrics.

**Stage 1 — Retrieval evaluation.** Ranking metrics on a synthetic labelled test set:

| Metric | Definition |
|---|---|
| Hit Rate@K | Is the ground-truth chunk in the top-K retrieved results? |
| MRR (Mean Reciprocal Rank) | How high did the ground-truth chunk rank on average? |
| nDCG@K (Normalised Discounted Cumulative Gain) | Position-weighted relevance of the retrieved ranking |

**Stage 2 — Generation evaluation.** The **RAG Triad** plus semantic metrics:

| Metric | Axis | What it measures |
|---|---|---|
| Context Relevance | Query → Context | Did retrieval find the right chunks? |
| Faithfulness | Context → Answer | Is every claim supported by the context? |
| Answer Relevance | Query → Answer | Did the answer address the question? |
| Perplexity | Fluency | Does the text look natural? |
| BERTScore | Semantic | How close is the answer to a reference? |
| NLI (Natural Language Inference) | Entailment | Does the context logically entail the answer? |

The RAG Triad is the key contribution: it isolates three distinct failure modes (retrieval failure, generation failure, relevance failure) that a single pass/fail score cannot distinguish.

### 4.2 Notebook Reference

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

### 4.3 Results & Metrics

The evaluation protocol is fully reproducible: every notebook loads the same datasets, uses the same embedding model and LLM, and computes the same metrics. Cross-notebook comparisons are therefore meaningful.

Key qualitative findings from the survey:

- **Chunking is the highest-leverage design choice.** `chunk_size=1000` with `chunk_overlap=200` produced the best balance between precision and semantic coherence on arXiv abstracts; smaller chunks improved precision but broke multi-sentence reasoning.
- **Retrieval strategy should be matched to the query distribution.** Similarity search dominates when queries are specific; MMR dominates when the corpus has many near-duplicate chunks (e.g., Drake lyrics).
- **Distance metric matters less than expected.** On unit-normalised embeddings, cosine and inner product produce near-identical rankings; L2 diverges only when embeddings are not normalised.
- **Prompt template changes the answer more than the retriever does.** Adding chain-of-thought scaffolding produced larger gains in answer relevance than swapping from top-3 similarity to top-5 MMR.
- **The RAG Triad isolates failure modes.** A generation that looked "wrong" under BLEU often turned out to be a faithfulness failure rather than a retrieval failure — a distinction that a single score would have obscured.

These findings are illustrative rather than universal — the survey is designed to make the trade-offs measurable, not to prescribe a single configuration.

---

## 5. Installation and Dataset Setup

### 5.1 Prerequisites

- **Python 3.10+**
- **~30 GB free disk** (HuggingFace model weights)
- **GPU with ≥ 8 GB VRAM** for Llama-3.1-8B inference (4-bit quantised). CPU works but is very slow.
- A **HuggingFace token** with read scope. Accept the Llama license at `meta-llama/Llama-3.1-8B-Instruct`.

### 5.2 Installation Steps

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

**Colab users** — `make` is not preinstalled. Run `!apt-get -qq install -y make` once per session.

### 5.3 Environment File

```bash
cp .env.example .env
```

Edit `.env`:

```
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

### 5.4 Dataset Setup

This repository uses **three Kaggle datasets**. They are downloaded into `data/` at the repo root. The download commands below work in Colab and locally.

**Step 1 — Configure Kaggle credentials**

Download `kaggle.json` from https://www.kaggle.com/settings/account → **Create New API Token**. Then:

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

**Step 2 — Download the datasets**

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

**Step 3 — Verify `data/`**

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

**Column note.** The arXiv CSV used here has columns `titles`, `summaries`, `terms`. The default config sets `content_columns: [summaries]`. If you swap in the original `spsayakpaul` release with an `abstracts` column, change the config accordingly.

---

## 6. Running Experiments

There are two ways to run the pipeline: as notebooks (for exploration) or as a script (for reproducibility).

### 6.1 Option A — Notebooks (recommended for first-timers)

```bash
# 1. Convert jupytext scripts to real .ipynb
pip install -q jupytext
jupytext --to notebook notebooks/*.py

# 2. Launch Jupyter
jupyter lab
```

Inside Jupyter, run the notebooks in order:

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

`03_embeddings` can be run any time — it only compares embedding providers.

### 6.2 Option B — Scripts (recommended for reproducibility)

Build the index from the CLI:

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

Load and query from Python:

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

### 6.3 Colab Quickstart

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

## 7. Configuration Reference

Every parameter that affects the pipeline lives in `configs/default.yaml`:

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

**Secrets (API keys, tokens) are never in the YAML** — they come from `.env`:

```
HUGGINGFACEHUB_API_TOKEN=hf_...
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

---

## 8. Extending the Pipeline

The factory-based design makes it straightforward to add new stages or swap existing ones:

- **New loader** — add a class under `src/rag_pipeline/ingestion/loaders/` and register it in the loader factory. A `type` value in the YAML config will then select it.
- **New splitter** — add a class under `src/rag_pipeline/splitting/` and register it. A `type` value selects it.
- **New embedding provider** — add a class under `src/rag_pipeline/embeddings/providers/` and register it. A `provider` value selects it.
- **New vector store** — add a class under `src/rag_pipeline/vectorstores/` and register it. A `type` value selects it.
- **New retriever** — add a builder under `src/rag_pipeline/retrieval/` and register it. A `search_type` value selects it.
- **New prompt template** — add it to `src/rag_pipeline/generation/prompts.py` and register it in the template registry.
- **New evaluation metric** — add it to `src/rag_pipeline/evaluation/` and register it in the metrics registry.

Each extension is a matter of adding a class and a registration line. The rest of the pipeline — ingestion, indexing, retrieval, generation, evaluation — remains unchanged.

---

## 9. Conclusion

This repository provides a **reproducible empirical survey** of the Retrieval-Augmented Generation pipeline, covering eight stages across their full design spaces: four loader families, six splitter strategies, four embedding providers, two vector stores, four retrieval strategies, nine prompt templates, and eleven evaluation metrics. By fixing the datasets, embedding model, LLM, and evaluation protocol — and by exposing every design choice as a YAML configuration knob — the survey makes the RAG design space legible and measurable.

Key observations from the survey:

- **Chunking is the highest-leverage design choice.** A single value of `chunk_size` affects retrieval precision, semantic coherence, and prompt budget simultaneously, and no downstream stage can fully recover from a poor choice.
- **Distance metric matters less than expected** when embeddings are unit-normalised: cosine and inner product produce near-identical rankings, and L2 diverges only when normalisation is skipped.
- **Prompt template dominates retriever configuration** for answer relevance on factual Q&A. Chain-of-thought scaffolding in the prompt produced larger gains than increasing top-K or switching retrieval strategy.
- **The RAG Triad is necessary for diagnosis.** A single pass/fail score cannot distinguish retrieval failure from generation failure from relevance failure; the Triad isolates all three.
- **Reproducibility requires discipline at the boundary.** YAML-driven orchestration, secrets separated from configuration, sampled data with fixed seeds, and a uniform `RAGPipeline` facade are all necessary conditions for the survey to be re-runnable on a different machine.

The repository is designed as a **research artefact**: every stage is documented at the level of design rationale and failure mode, every experiment is driven by a version-controlled configuration file, and every comparison is exercised through a reproducible notebook. It is intended to be forked, extended with new loaders/splitters/providers/stores/retrievers/prompts/metrics, and reused as a baseline for future work in the RAG design space.

---

## 10. References

If you use this repository in research, please cite:

```bibtex
@misc{ragsurvey2025,
  title  = {A Survey of Retrieval-Augmented Generation: Modular Pipeline and Experiments},
  author = {MisterFOURXXX},
  year   = {2025},
  url    = {https://github.com/MisterFOURXXX/a-survey-of-retrieval-augmented-generation-RAG}
}
```

**Foundational literature**

1. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401
2. Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* arXiv:2004.04906
3. Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997
4. Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217

**Tools and frameworks**

- LangChain documentation — https://python.langchain.com/
- Chroma documentation — https://docs.trychroma.com/
- FAISS documentation — https://faiss.ai/

Additional resources referenced by the original project:

- Hugging Face `transformers` and `sentence-transformers` documentation for embedding models.
- `evaluate`, `bert-score`, and `sentence-transformers` for evaluation metrics.
- Kaggle documentation for dataset download.

---

## 11. License

MIT — see `LICENSE`.

**Acknowledgements and licensing notes.**

- **Code:** MIT License.
- **Datasets:** Each dataset is distributed under its own license on Kaggle. Respect the attribution requirements of `snehaanbhawal/resume-dataset`, `juicobowley/drake-lyrics`, and `spsayakpaul/arxiv-paper-abstracts`.
- **Base models:** `meta-llama/Llama-3.1-8B-Instruct` is distributed under the Llama 3.1 Community License. `BAAI/bge-small-en-v1.5` is distributed under MIT.

Provided for academic and educational research use. If you fork this repository, preserve the attribution and licensing sections that accompany the datasets and models.

---

## 12. About

This repository is maintained as a **reproducible academic research artefact** and a **systems-level reference** for the Retrieval-Augmented Generation design space. It is intended to support:

- **Comparative empirical studies** of RAG design choices on a common footing.
- **Educational use** as a working codebase that operationalises the ingestion, chunking, embedding, vector-store, retrieval, generation, memory, and evaluation concepts documented in the Foundations chapter.
- **Extension** with new loaders, splitters, embedding providers, vector stores, retrievers, prompt templates, and evaluation metrics through the factory + YAML configuration system.
- **Reproducibility audits** of the surveyed RAG configurations, using the pinned dependency stack and the shared evaluation protocol.

It is designed as the **retrieval-augmented generation companion** to the broader LLM engineering survey series (Architectures, Fine-Tuning Optimization, Fine-Tuning Approaches, LLMOps). Together, these repositories form a coherent reference for the architecture, training, and retrieval sides of modern LLM engineering.

The repository is designed to be forked, extended, and adapted to new domains. It is a **survey as an artefact**: not only a summary of the field, but a working baseline that the field can build on.
