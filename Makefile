.PHONY: help install hooks pre-commit lock test lint format check run api pipeline docker-up docker-down eval eval-validate

UV ?= uv
EVAL_ARGS ?=

help:
	@echo "hooks          Install pre-commit Git hooks"
	@echo "pre-commit     Run all commit checks"
	@echo "eval           Evaluate the local API; EVAL_ARGS passes runner options"
	@echo "eval-validate  Validate fixtures without model calls"
	@echo "install        Install runtime and development dependencies"
	@echo "lock           Update uv.lock"
	@echo "test           Run unit tests"
	@echo "lint           Check Python code with Ruff"
	@echo "format         Format Python code with Ruff"
	@echo "check          Run lint and tests"
	@echo "run            Start Gradio web UI"
	@echo "api            Start FastAPI"
	@echo "pipeline       Run one multimodal pipeline cycle"
	@echo "docker-up      Build and start Compose services"
	@echo "docker-down    Stop Compose services"

install:
	$(UV) sync --locked
	$(UV) run --locked pre-commit install

hooks:
	$(UV) run --locked pre-commit install

pre-commit:
	$(UV) run --locked pre-commit run --all-files

lock:
	$(UV) lock

test:
	$(UV) run --locked pytest -q

lint:
	$(UV) run --locked ruff check src tests scripts

format:
	$(UV) run --locked ruff format src tests scripts

check: lint test

run:
	$(UV) run --locked python -m src.web_ui

api:
	$(UV) run --locked python -m src.api

pipeline:
	$(UV) run --locked python -m src.bot_pipeline

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

eval:
	$(UV) run --locked python -m scripts.run_evals $(EVAL_ARGS)

eval-validate:
	$(UV) run --locked python -m scripts.run_evals --validate-only
