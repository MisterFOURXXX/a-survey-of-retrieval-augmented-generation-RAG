from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field

class Config(BaseModel):
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
    def from_yaml(cls, path):
        with open(path) as fh:
            return cls(**(yaml.safe_load(fh) or {}))

def load_config(path=None):
    return Config() if path is None else Config.from_yaml(path)
