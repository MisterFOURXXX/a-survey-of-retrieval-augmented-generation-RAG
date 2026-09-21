"""Metadata enrichment utilities."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document


def _file_type(source: str) -> Optional[str]:
    if not source:
        return None
    return Path(source).suffix.lstrip(".").lower() or None


def enrich_metadata(
    documents: List[Document],
    extra: Optional[dict] = None,
    hash_content: bool = True,
    add_word_count: bool = True,
) -> List[Document]:
    """Attach processed_at / content_hash / file_type / word_count metadata."""
    now = datetime.now(timezone.utc).isoformat()
    for doc in documents:
        doc.metadata["processed_at"] = now
        if hash_content:
            doc.metadata["content_hash"] = hashlib.md5(
                doc.page_content.encode("utf-8")
            ).hexdigest()
        src = doc.metadata.get("source")
        if src:
            ft = _file_type(src)
            if ft:
                doc.metadata["file_type"] = ft
        if add_word_count:
            doc.metadata["word_count"] = len(doc.page_content.split())
        if extra:
            doc.metadata.update(extra)
    return documents
