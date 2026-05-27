# Stage 1: Build virtual environment
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for extremely fast and locked dependency synchronization
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

# Synchronize dependencies (excluding dev group for a smaller image)
RUN uv sync --no-dev --frozen

# Stage 2: Final runtime image
FROM python:3.12-slim

# Install system dependencies required for XGBoost/LightGBM (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built virtual environment and application source code
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src
COPY api/ ./api
COPY models/ ./models

# Set PATH and PYTHONPATH to use the synchronized venv and project modules
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI server using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
