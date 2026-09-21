from __future__ import annotations
import re
from typing import List, Optional, Sequence
from langchain_core.documents import Document
from rag_pipeline.utils import logger

DEFAULT_REMOVE_PATTERNS = [r"Page \d+ of \d+", r"Copyright (c).*", r"All rights reserved.*"]

def clean_text(text, remove_patterns=None, collapse_whitespace=True):
    patterns = list(remove_patterns) if remove_patterns else DEFAULT_REMOVE_PATTERNS
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_and_filter_documents(documents, min_length=50, max_length=10000, remove_patterns=None):
    cleaned = []
    for doc in documents:
        c = clean_text(doc.page_content, remove_patterns=remove_patterns)
        if min_length <= len(c) <= max_length:
            doc.page_content = c
            cleaned.append(doc)
    logger.info("Cleaned %d -> %d documents", len(documents), len(cleaned))
    return cleaned
