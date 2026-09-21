"""Document loaders for all supported source types."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document

from rag_pipeline.utils import logger


# --------------------------------------------------------------------------
# Individual loaders
# --------------------------------------------------------------------------

def load_text(path: str | Path, encoding: str = "utf-8", **kwargs: Any) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    return TextLoader(str(path), encoding=encoding, **kwargs).load()


def load_directory(
    path: str | Path,
    glob: str = "**/*.txt",
    loader_cls: str = "text",
    encoding: str = "utf-8",
    **kwargs: Any,
) -> List[Document]:
    from langchain_community.document_loaders import DirectoryLoader
    loader_map = {
        "text": "langchain_community.document_loaders.TextLoader",
        "pdf": "langchain_community.document_loaders.PyPDFLoader",
    }
    if loader_cls not in loader_map:
        raise ValueError(f"Unsupported directory loader: {loader_cls}")
    loader = DirectoryLoader(
        str(path),
        glob=glob,
        loader_cls=loader_map[loader_cls],
        loader_kwargs={"encoding": encoding} if loader_cls == "text" else {},
        **kwargs,
    )
    return loader.load()


def load_csv(
    path: str | Path,
    content_columns: Optional[List[str]] = None,
    metadata_columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Document]:
    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(
        file_path=str(path),
        content_columns=content_columns,
        metadata_columns=metadata_columns,
        **kwargs,
    )
    return loader.load()


def load_pdf(path: str | Path, **kwargs: Any) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(str(path), **kwargs).load()


def load_json(
    path: str | Path,
    jq_schema: str = ".[]",
    text_content: bool = False,
    **kwargs: Any,
) -> List[Document]:
    from langchain_community.document_loaders import JSONLoader
    loader = JSONLoader(
        file_path=str(path),
        jq_schema=jq_schema,
        text_content=text_content,
        **kwargs,
    )
    return loader.load()


def load_web(
    urls: List[str],
    parse_only_classes: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Document]:
    import bs4
    from langchain_community.document_loaders import WebBaseLoader

    bs_kwargs: Dict[str, Any] = {}
    if parse_only_classes:
        bs_kwargs["parse_only"] = bs4.SoupStrainer(class_=tuple(parse_only_classes))

    loader = WebBaseLoader(web_paths=urls, bs_kwargs=bs_kwargs or None, **kwargs)
    return loader.load()


def load_sql(engine_url: str, query: str, **kwargs: Any) -> List[Document]:
    from langchain_community.document_loaders import SQLDatabaseLoader
    from sqlalchemy import create_engine

    engine = create_engine(engine_url)
    return SQLDatabaseLoader(engine, query, **kwargs).load()


def load_s3(
    bucket: str,
    key: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    **kwargs: Any,
) -> List[Document]:
    from langchain_community.document_loaders import S3FileLoader

    loader = S3FileLoader(
        bucket=bucket,
        key=key,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        **kwargs,
    )
    return loader.load()


def load_git(
    clone_url: str,
    repo_path: str,
    file_filter: Optional[Any] = None,
    **kwargs: Any,
) -> List[Document]:
    from langchain_community.document_loaders import GitLoader

    loader = GitLoader(
        clone_url=clone_url,
        repo_path=repo_path,
        file_filter=file_filter,
        **kwargs,
    )
    return loader.load()


# --------------------------------------------------------------------------
# Dispatcher + lazy loading
# --------------------------------------------------------------------------

_LOADER_DISPATCH = {
    "text": load_text,
    "directory": load_directory,
    "csv": load_csv,
    "pdf": load_pdf,
    "json": load_json,
    "web": load_web,
    "sql": load_sql,
    "s3": load_s3,
    "git": load_git,
}


def load_source(spec: Dict[str, Any]) -> List[Document]:
    """Load a single source spec from the config."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in _LOADER_DISPATCH:
        raise ValueError(
            f"Unknown loader type: {kind}. Available: {list(_LOADER_DISPATCH)}"
        )
    logger.info("Loading source type=%s spec=%s", kind, spec)
    return _LOADER_DISPATCH[kind](**spec)


def load_documents(sources: List[Dict[str, Any]]) -> List[Document]:
    """Load and concatenate all sources."""
    docs: List[Document] = []
    for spec in sources:
        docs.extend(load_source(spec))
    logger.info("Loaded %d documents from %d source(s)", len(docs), len(sources))
    return docs


def lazy_load_source(spec: Dict[str, Any]) -> Iterator[Document]:
    """Generator-based lazy loading for very large sources."""
    yield from load_source(spec)
