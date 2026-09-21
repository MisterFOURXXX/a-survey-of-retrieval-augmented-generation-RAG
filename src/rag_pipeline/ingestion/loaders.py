"""Document loaders for all supported source types."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document

from rag_pipeline.utils import logger


# --------------------------------------------------------------------------
# Individual loaders
# --------------------------------------------------------------------------

def load_text(path, encoding="utf-8", **kw):
    from langchain_community.document_loaders import TextLoader
    return TextLoader(str(path), encoding=encoding, **kw).load()


def load_directory(path, glob="**/*.txt", loader_cls="text", encoding="utf-8", **kw):
    from langchain_community.document_loaders import DirectoryLoader
    loader_map = {
        "text": "langchain_community.document_loaders.TextLoader",
        "pdf":  "langchain_community.document_loaders.PyPDFLoader",
    }
    if loader_cls not in loader_map:
        raise ValueError(f"Unsupported directory loader: {loader_cls}")
    loader = DirectoryLoader(
        str(path), glob=glob, loader_cls=loader_map[loader_cls],
        loader_kwargs={"encoding": encoding} if loader_cls == "text" else {},
        **kw,
    )
    return loader.load()


def load_csv(path, content_columns=None, metadata_columns=None, **kw):
    """Load a CSV.

    Only forwards content_columns / metadata_columns when set. LangChain's
    CSVLoader iterates over metadata_columns unconditionally and crashes
    when it is None.
    """
    from langchain_community.document_loaders.csv_loader import CSVLoader
    kwargs = {"file_path": str(path)}
    if content_columns is not None:
        kwargs["content_columns"] = content_columns
    if metadata_columns is not None:
        kwargs["metadata_columns"] = metadata_columns
    kwargs.update(kw)
    return CSVLoader(**kwargs).load()


def load_pdf(path, **kw):
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(str(path), **kw).load()


def load_json(path, jq_schema=".[]", text_content=False, **kw):
    from langchain_community.document_loaders import JSONLoader
    return JSONLoader(
        file_path=str(path), jq_schema=jq_schema,
        text_content=text_content, **kw,
    ).load()


def load_web(urls, parse_only_classes=None, **kw):
    import bs4
    from langchain_community.document_loaders import WebBaseLoader
    bk = {}
    if parse_only_classes:
        bk["parse_only"] = bs4.SoupStrainer(class_=tuple(parse_only_classes))
    return WebBaseLoader(web_paths=urls, bs_kwargs=bk or None, **kw).load()


def load_web_stream(urls, parse_only_classes=None, **kw):
    """Lazy-load web pages one at a time."""
    import bs4
    from langchain_community.document_loaders import WebBaseLoader
    bk = {}
    if parse_only_classes:
        bk["parse_only"] = bs4.SoupStrainer(class_=tuple(parse_only_classes))
    loader = WebBaseLoader(web_paths=urls, bs_kwargs=bk or None, **kw)
    return list(loader.lazy_load())


def load_sql(engine_url, query, **kw):
    from langchain_community.document_loaders import SQLDatabaseLoader
    from sqlalchemy import create_engine
    engine = create_engine(engine_url)
    return SQLDatabaseLoader(engine, query, **kw).load()


def load_s3(bucket, key, aws_access_key_id=None, aws_secret_access_key=None, **kw):
    from langchain_community.document_loaders import S3FileLoader
    return S3FileLoader(
        bucket=bucket, key=key,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        **kw,
    ).load()


def load_git(clone_url, repo_path, file_filter=None, **kw):
    from langchain_community.document_loaders import GitLoader
    return GitLoader(
        clone_url=clone_url, repo_path=repo_path,
        file_filter=file_filter, **kw,
    ).load()


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

_LOADER_DISPATCH = {
    "text":       load_text,
    "directory":  load_directory,
    "csv":        load_csv,
    "pdf":        load_pdf,
    "json":       load_json,
    "web":        load_web,
    "web_stream": load_web_stream,
    "sql":        load_sql,
    "s3":         load_s3,
    "git":        load_git,
}


def load_source(spec):
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in _LOADER_DISPATCH:
        raise ValueError(
            f"Unknown loader type: {kind}. Available: {list(_LOADER_DISPATCH)}"
        )
    logger.info("Loading source type=%s spec=%s", kind, spec)
    return _LOADER_DISPATCH[kind](**spec)


def _sample_documents(
    docs: List[Document],
    fraction: Optional[float],
    size: Optional[int],
    seed: int = 42,
) -> List[Document]:
    """Randomly sample documents.

    Priority:
      - If `size` is set, keep exactly `size` documents (capped at len(docs)).
      - Else if `fraction` is set (0 < fraction <= 1), keep round(fraction * len(docs)).
      - Else return docs unchanged.

    Uses `random.Random(seed)` so results are reproducible across runs.
    """
    total = len(docs)
    if total == 0:
        return docs

    if size is not None:
        n = min(int(size), total)
    elif fraction is not None:
        if not (0 < float(fraction) <= 1):
            raise ValueError(f"sample_fraction must be in (0, 1], got {fraction}")
        n = max(1, int(round(float(fraction) * total)))
    else:
        return docs

    if n >= total:
        logger.info("Sampling requested but n=%d >= total=%d — keeping all.", n, total)
        return docs

    rng = random.Random(seed)
    sampled = rng.sample(docs, n)
    logger.info("Sampled %d / %d documents (fraction=%s, size=%s, seed=%d)",
                n, total, fraction, size, seed)
    return sampled


def load_documents(sources, sample_fraction=None, sample_size=None,
                   sample_seed=42):
    """Load and concatenate all sources, then optionally sample.

    Args:
        sources:        List of source specs (see `_LOADER_DISPATCH`).
        sample_fraction: Keep this fraction of the *combined* corpus (0 < f <= 1).
        sample_size:    Keep exactly this many documents (overrides fraction).
        sample_seed:    RNG seed for reproducibility.
    """
    docs: List[Document] = []
    for s in sources:
        docs.extend(load_source(s))
    logger.info("Loaded %d documents from %d source(s)", len(docs), len(sources))

    docs = _sample_documents(docs, sample_fraction, sample_size, sample_seed)
    return docs


def lazy_load_source(spec):
    yield from load_source(spec)