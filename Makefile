.PHONY: install dev build-index load-index notebooks test lint format clean help

help:
	@echo "Targets:"
	@echo "  install      Install runtime dependencies"
	@echo "  dev          Install dev dependencies + pre-commit hooks"
	@echo "  build-index  Build the vector index from configs/default.yaml"
	@echo "  load-index   Load an existing index (smoke test)"
	@echo "  notebooks    Convert jupytext scripts into .ipynb notebooks"
	@echo "  test         Run the smoke test suite"
	@echo "  lint         Ruff + black check"
	@echo "  format       Ruff + black fix"
	@echo "  clean        Remove caches and build artifacts"

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

dev:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e ".[dev]"
	pre-commit install

build-index:
	python scripts/build_index.py --config configs/default.yaml

load-index:
	python scripts/build_index.py --config configs/default.yaml --load-only

notebooks:
	jupytext --to notebook notebooks/*.py

test:
	pytest -q

lint:
	ruff check src tests scripts
	black --check src tests scripts

format:
	ruff check --fix src tests scripts
	black src tests scripts

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
