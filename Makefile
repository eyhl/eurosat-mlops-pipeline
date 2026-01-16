.PHONY: help venv setup fmt lint test train eval predict clean

PY ?= python
CONFIG ?= configs/default.yaml

help:
	@echo "Targets:"
	@echo "  make venv      - create .venv (uv if available, else python -m venv)"
	@echo "  make setup     - install deps (editable + dev)"
	@echo "  make fmt       - format code (ruff)"
	@echo "  make lint      - lint code (ruff)"
	@echo "  make test      - run pytest"
	@echo "  make train     - train using CONFIG=$(CONFIG)"
	@echo "  make eval      - evaluate RUN=artifacts/<run_id>"
	@echo "  make predict   - predict RUN=... INPUT=... OUTPUT=predictions.csv"

venv:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv; \
	else \
		$(PY) -m venv .venv; \
	fi

setup: venv
	@if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi; \
	python -m pip install -U pip; \
	python -m pip install -e ".[dev]"

fmt:
	@if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi; \
	ruff format .

lint:
	@if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi; \
	ruff check .

test:
	@if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi; \
	pytest

train:
	@if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi; \
	python -m src.train --config $(CONFIG)

eval:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=artifacts/<run_id>"; exit 2; fi; \
	if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi; \
	python -m src.evaluate --run $(RUN)

predict:
	@if [ -z "$(RUN)" ]; then echo "Missing RUN=artifacts/<run_id>"; exit 2; fi; \
	if [ -z "$(INPUT)" ]; then echo "Missing INPUT=<folder>"; exit 2; fi; \
	if [ -z "$(OUTPUT)" ]; then echo "Missing OUTPUT=predictions.csv"; exit 2; fi; \
	if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi; \
	python -m src.predict --run $(RUN) --input $(INPUT) --output $(OUTPUT)

clean:
	rm -rf .pytest_cache .ruff_cache
