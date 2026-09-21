"""Shared helpers: env loading, logging, lazy imports."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List

from langchain_core.documents import Document

logger = logging.getLogger("rag_pipeline")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_env(path: str | Path = ".env") -> None:
    """Load .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = Path(path)
    if p.exists():
        load_dotenv(p)


def get_secret(name: str, default: str | None = None) -> str | None:
    """Fetch a secret from env; supports Kaggle secrets as a fallback."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret(name)
    except Exception:
        return default


def format_docs(docs: Iterable[Document], max_chars: int | None = None) -> str:
    """Format documents as `[Row X] content` blocks for prompts."""
    lines: List[str] = []
    for i, doc in enumerate(docs, 1):
        row = doc.metadata.get("row", i)
        content = doc.page_content.strip()
        if max_chars is not None:
            content = content[:max_chars]
        lines.append(f"[Row {row}] {content}")
    return "\n\n".join(lines) if lines else "(No relevant documents retrieved)"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def batch_iter(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
