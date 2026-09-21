"""CLI: build (or rebuild) the RAG index from a config file."""
from __future__ import annotations

import argparse

from rag_pipeline.config import load_config
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.utils import load_env, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RAG index.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load an existing index instead of building a new one.",
    )
    args = parser.parse_args()

    load_env()
    setup_logging()

    cfg = load_config(args.config)
    pipeline = RAGPipeline(cfg)

    if args.load_only:
        pipeline.load_index()
        print("Index loaded.")
    else:
        pipeline.build_index()
        print("Index built and persisted.")


if __name__ == "__main__":
    main()
