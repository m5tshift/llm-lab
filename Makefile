.PHONY: help install install-eval hooks pre-commit lock test lint format check run api pipeline docker-up docker-down eval-text eval-audio eval-image eval-video eval-llm eval-llm-review summary-table report-pdf

UV ?= uv
EVAL_ARGS ?=

help:
	@echo "hooks          Install pre-commit Git hooks"
	@echo "pre-commit     Run all commit checks"
	@echo "install-eval   Install dependencies including the eval group"
	@echo "eval-text      Task 1: emotions on CEDR test (EVAL_ARGS passes options)"
	@echo "eval-audio     Task 2: WER/RTF on FLEURS ru test"
	@echo "eval-image     Task 3: scene classification on labeled frames"
	@echo "eval-video     Task 4: mAP on labeled frames and tracking on clips"
	@echo "eval-llm       Task 5: LLM answers on scenarios, writes a review CSV"
	@echo "eval-llm-review  Merge manual LLM scores: REVIEW=results/llm/<file>.review.csv"
	@echo "summary-table  Print the report summary table from the latest results"
	@echo "report-pdf     Build build/REPORT.pdf from REPORT.md (needs Chrome)"
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

install-eval:
	$(UV) sync --locked --group eval

hooks:
	$(UV) run --locked pre-commit install

pre-commit:
	$(UV) run --locked pre-commit run --all-files

lock:
	$(UV) lock

test:
	$(UV) run --locked pytest -q

lint:
	$(UV) run --locked ruff check src tests scripts evals

format:
	$(UV) run --locked ruff format src tests scripts evals

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

EVAL = $(UV) run --locked --group eval python -m

eval-text:
	$(EVAL) evals.text $(EVAL_ARGS)

eval-audio:
	$(EVAL) evals.audio $(EVAL_ARGS)

eval-image:
	$(EVAL) evals.image --manifest evals/data/scene_labels.csv $(EVAL_ARGS)

eval-video:
	$(EVAL) evals.video --det-dir evals/local/video/det --clips evals/data/video_clips.csv $(EVAL_ARGS)

eval-llm:
	$(EVAL) evals.llm $(EVAL_ARGS)

eval-llm-review:
	$(EVAL) evals.llm --review $(REVIEW)

summary-table:
	$(UV) run --locked python -m scripts.summary_table

report-pdf:
	$(UV) run --locked python -m scripts.build_pdf $(PDF_ARGS)
