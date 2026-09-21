"""Configuration loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Top-level configuration for the RAG pipeline.

    Values are kept as plain dicts/lists so the config file can evolve without
    requiring schema migrations. Sub-packages pull only what they need.
    """

    data: Dict[str, Any] = Field(default_factory=dict)
    cleaning: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    splitting: Dict[str, Any] = Field(default_factory=dict)
    embeddings: Dict[str, Any] = Field(default_factory=dict)
    vectorstore: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    llm: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(), fh, sort_keys=False)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load config from YAML or return defaults."""
    if path is None:
        return Config()
    return Config.from_yaml(path)
