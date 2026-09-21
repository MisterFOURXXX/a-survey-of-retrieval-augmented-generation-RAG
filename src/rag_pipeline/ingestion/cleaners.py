"""Text cleaning and quality filtering."""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from langchain_core.documents import Document

from rag_pipeline.utils import logger

DEFAULT_REMOVE_PATTERNS = [
    r"Page \d+ of \d+",
    r"Copyright (c).*",
    r"All rights reserved.*",
]


def clean_text(
    text: str,
    remove_patterns: Optional[Sequence[str]] = None,
    collapse_whitespace: bool = True,
) -> str:
    patterns = list(remove_patterns) if remove_patterns else DEFAULT_REMOVE_PATTERNS
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_and_filter_documents(
    documents: List[Document],
    min_length: int = 50,
    max_length: int = 10_000,
    remove_patterns: Optional[List[str]] = None,
) -> List[Document]:
    """Clean each document and drop those outside the length window."""
    cleaned: List[Document] = []
    for doc in documents:
        content = clean_text(doc.page_content, remove_patterns=remove_patterns)
        if min_length <= len(content) <= max_length:
            doc.page_content = content
            cleaned.append(doc)
    logger.info("Cleaned %d -> %d documents", len(documents), len(cleaned))
    return cleaned
