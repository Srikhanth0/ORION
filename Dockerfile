# ═══════════════════════════════════════════════════════════════
# ORION — Multi-stage Dockerfile
# ═══════════════════════════════════════════════════════════════
# Stage 1: Builder — install all deps including dev tools
# Stage 2: Runtime — copy only site-packages + app code
# Final image < 600 MB. No build tools in runtime.

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install all dependencies
RUN uv sync --frozen --no-dev --no-editable

# Copy application code
COPY orion/ orion/
COPY configs/ configs/
COPY prompts/ prompts/
COPY scripts/ scripts/

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install minimal runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r orion && useradd -r -g orion orion

# Copy virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY --from=builder /build/orion /app/orion
COPY --from=builder /build/configs /app/configs
COPY --from=builder /build/prompts /app/prompts
COPY --from=builder /build/scripts /app/scripts

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run as non-root
USER orion

EXPOSE 8080 9091

CMD ["python", "-m", "uvicorn", "orion.api.server:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-keep-alive", "30"]
