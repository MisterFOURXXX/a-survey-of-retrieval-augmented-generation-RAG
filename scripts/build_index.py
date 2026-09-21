from __future__ import annotations
import argparse
from rag_pipeline.config import load_config
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.utils import load_env, setup_logging

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--load-only", action="store_true")
    args = p.parse_args()
    load_env(); setup_logging()
    pipe = RAGPipeline(load_config(args.config))
    if args.load_only:
        pipe.load_index(); print("Index loaded.")
    else:
        pipe.build_index(); print("Index built and persisted.")

if __name__ == "__main__":
    main()
