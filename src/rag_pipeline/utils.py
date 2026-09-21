"""Shared helpers: env loading, logging, secret lookup, doc formatting."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger("rag_pipeline")


# ---------------------------------------------------------------------------
# Logging + env
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def load_env(path: str | Path = ".env", override: bool = False) -> None:
    """Load a .env file into os.environ (no-op if python-dotenv is missing)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = Path(path)
    if p.exists():
        load_dotenv(p, override=override)


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return a secret from env or Kaggle secrets (best effort)."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret(name)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------

def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk upward from `start` (or cwd) until `configs/` and `src/` exist."""
    p = Path(start).resolve() if start else Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return p


def load_notebook_config(
    config_path: str | Path = "configs/default.yaml",
    env_path: str | Path = ".env",
) -> Tuple[Any, Path]:
    """Load config + secrets for notebooks.

    Returns
    -------
    (Config, repo_root)

    Usage in a notebook::

        from rag_pipeline.utils import load_notebook_config
        cfg, REPO = load_notebook_config()
        faiss_dir = REPO / cfg.paths["faiss_index"]
    """
    from rag_pipeline.config import load_config

    repo = find_repo_root()

    # Load .env if present (does not override existing env vars by default)
    env_file = repo / env_path
    load_env(env_file, override=False)

    cfg_file = repo / config_path
    cfg = load_config(cfg_file) if cfg_file.exists() else load_config()
    return cfg, repo


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def format_docs(docs: Iterable[Document], max_chars: int | None = None) -> str:
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
