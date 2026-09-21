"""High-level RAG pipeline facade."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document

from rag_pipeline.config import Config, load_config
from rag_pipeline.embeddings import build_embeddings
from rag_pipeline.generation import RAG_PROMPT_TEMPLATE, build_llms, get_prompt
from rag_pipeline.ingestion import (
    clean_and_filter_documents, enrich_metadata, load_documents,
)
from rag_pipeline.memory import ConversationMemory
from rag_pipeline.retrieval import build_retriever
from rag_pipeline.splitting import split_documents
from rag_pipeline.utils import format_docs
from rag_pipeline.vectorstores import build_vectorstore, load_vectorstore


class RAGPipeline:
    """End-to-end RAG pipeline driven by a Config."""

    def __init__(self, config: Optional[Config | str | Path] = None):
        if config is None:
            self.config = Config()
        elif isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config

        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llms: Dict[str, Any] = {}
        self.memory = ConversationMemory()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def build_embeddings(self):
        self.embeddings = build_embeddings(self.config.embeddings)
        return self.embeddings

    def load_and_split(self) -> List[Document]:
        docs = load_documents(self.config.data.get("sources", []))
        if self.config.cleaning:
            docs = clean_and_filter_documents(docs, **self.config.cleaning)
        if self.config.metadata.get("enrich"):
            docs = enrich_metadata(docs)

        split_cfg = dict(self.config.splitting)
        if split_cfg.get("type") == "semantic":
            if self.embeddings is None:
                self.build_embeddings()
            split_cfg["_embeddings"] = self.embeddings
        return split_documents(docs, split_cfg)

    def build_index(self) -> None:
        if self.embeddings is None:
            self.build_embeddings()
        chunks = self.load_and_split()
        self.vectorstore = build_vectorstore(
            chunks, self.embeddings, self.config.vectorstore
        )
        self.retriever = build_retriever(self.vectorstore, self.config.retrieval)

    def load_index(self) -> None:
        if self.embeddings is None:
            self.build_embeddings()
        self.vectorstore = load_vectorstore(self.embeddings, self.config.vectorstore)
        self.retriever = build_retriever(self.vectorstore, self.config.retrieval)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def build_llms(self):
        self.llms = build_llms(self.config.llm)
        return self.llms

    def _ensure(self):
        if self.retriever is None:
            raise RuntimeError("Call build_index() or load_index() first.")
        if not self.llms:
            self.build_llms()

    def retrieve(self, question: str) -> List[Document]:
        self._ensure()
        return self.retriever.invoke(question)

    def _format_prompt(self, question: str, docs: List[Document],
                       prompt_name: str = "rag",
                       track_memory: bool = False) -> str:
        template = get_prompt(prompt_name)
        context = format_docs(docs)

        if prompt_name == "conversation" or track_memory:
            return template.format(
                conversation_context=self.memory.get_context(),
                context=context,
                question=question,
            )
        return template.format(context=context, question=question)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------
    def query(self, question: str, model_name: Optional[str] = None,
              max_new_tokens: int = 200,
              prompt_name: str = "rag",
              track_memory: bool = False) -> str:
        self._ensure()
        docs = self.retrieve(question)
        prompt = self._format_prompt(question, docs, prompt_name, track_memory)

        if track_memory:
            self.memory.add_message("user", question)

        name = model_name or next(iter(self.llms))
        out = str(self.llms[name].invoke(prompt, max_new_tokens=max_new_tokens))

        if track_memory:
            self.memory.add_message("assistant", out[:300])
        return out

    def stream(self, question: str, model_name: Optional[str] = None,
               max_new_tokens: int = 200,
               prompt_name: str = "rag",
               track_memory: bool = False) -> Iterator[str]:
        self._ensure()
        docs = self.retrieve(question)
        prompt = self._format_prompt(question, docs, prompt_name, track_memory)

        if track_memory:
            self.memory.add_message("user", question)

        name = model_name or next(iter(self.llms))
        llm = self.llms[name]

        collected = ""
        for chunk in llm.stream(prompt, max_new_tokens=max_new_tokens):
            text = str(chunk)
            collected += text
            yield text

        if track_memory:
            self.memory.add_message("assistant", collected[:300])

    def batch_query(self, questions: List[str],
                    model_name: Optional[str] = None,
                    max_new_tokens: int = 200,
                    prompt_name: str = "rag") -> List[str]:
        """Batch-process multiple questions against a single model."""
        self._ensure()
        name = model_name or next(iter(self.llms))
        llm = self.llms[name]
        prompts = [self._format_prompt(q, self.retrieve(q), prompt_name)
                   for q in questions]
        return [str(r) for r in llm.batch(prompts, max_new_tokens=max_new_tokens)]

    def query_structured(self, question: str,
                         model_name: Optional[str] = None,
                         max_new_tokens: int = 250) -> str:
        """Generate a structured answer (Question / Direct answer / Summary ...)."""
        return self.query(
            question, model_name=model_name,
            max_new_tokens=max_new_tokens,
            prompt_name="structured",
        )
