from __future__ import annotations
import logging, os
from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document
logger = logging.getLogger("rag_pipeline")

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_env(path=".env"):
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if Path(path).exists():
        load_dotenv(path)

def get_secret(name, default=None):
    val = os.environ.get(name)
    if val: return val
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return default

def format_docs(docs, max_chars=None):
    lines = []
    for i, d in enumerate(docs, 1):
        row = d.metadata.get("row", i)
        content = d.page_content.strip()
        if max_chars is not None: content = content[:max_chars]
        lines.append(f"[Row {row}] {content}")
    return "\n\n".join(lines) if lines else "(No relevant documents retrieved)"

def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p
