.PHONY: install build-index load-index test clean help

help:
	@echo "Targets: install build-index load-index test clean"

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

build-index:
	python scripts/build_index.py --config configs/default.yaml

load-index:
	python scripts/build_index.py --config configs/default.yaml --load-only

test:
	pytest -q

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
