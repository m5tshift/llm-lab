# syntax=docker/dockerfile:1

# --- Stage 1: Build stage ---
FROM python:3.11-slim AS build

WORKDIR /build

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

# Install build dependencies if needed for compiling packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install locked runtime dependencies into a fixed path so console scripts remain valid
# when the environment is copied to the runtime stage.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project


# --- Stage 2: Production runtime stage ---
FROM python:3.11-slim AS runtime

# Install system runtime libraries required by OpenCV, PyTorch, Whisper/Librosa and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --ingroup appgroup --home /home/appuser appuser

# Prepare cache directories for model weights and configs
RUN mkdir -p /home/appuser/.cache/huggingface \
             /home/appuser/.cache/torch \
             /home/appuser/.config/Ultralytics \
             /app && \
    chown -R appuser:appgroup /home/appuser /app

# Copy virtual environment from build stage
COPY --from=build --chown=appuser:appgroup /opt/venv /opt/venv

WORKDIR /app

# Copy application source code
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup samples/ /app/samples/

# Configure runtime environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

USER appuser

# Expose Web UI (7860) and REST API (8000)
EXPOSE 7860 8000

# Default command starts the interactive Gradio Web UI
CMD ["python", "-m", "src.web_ui"]
