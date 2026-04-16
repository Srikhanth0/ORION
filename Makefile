# ═══════════════════════════════════════════════════════════════
# ORION — Development Makefile
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# ORION-FIX: Force UTF-8 for all Python subprocesses on Windows
export PYTHONUTF8 := 1

.PHONY: install dev test test-unit test-int lint lint-fix typecheck \
        podman-up podman-up-dev podman-down podman-build \
        eval health seed clean help

help: ## Show this help message
	@echo "ORION — Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────
install: ## Install all dependencies (prod + dev)
	@echo "──▶ Installing dependencies via uv..."
	uv sync --all-extras

# ── Development ──────────────────────────────────────────
dev: ## Start the FastAPI dev server with hot reload
	@echo "──▶ Starting ORION API server (dev mode)..."
	uv run uvicorn orion.api.server:app --reload --host 0.0.0.0 --port 8080

# ── Testing ──────────────────────────────────────────────
test: ## Run all tests (unit + integration)
	@echo "──▶ Running full test suite..."
	uv run pytest tests/ -v --tb=short

test-unit: ## Run only unit tests
	@echo "──▶ Running unit tests..."
	uv run pytest tests/unit/ -v --tb=short

test-int: ## Run only integration tests
	@echo "──▶ Running integration tests..."
	uv run pytest tests/integration/ -v --tb=short

# ── Code Quality ─────────────────────────────────────────
lint: ## Run ruff linter
	@echo "──▶ Linting with ruff..."
	uv run ruff check orion/ tests/ scripts/

lint-fix: ## Auto-fix lint errors and format
	@echo "──▶ Auto-fixing lint errors..."
	uv run ruff check --fix orion/ tests/ scripts/
	uv run ruff format orion/ tests/ scripts/

typecheck: ## Run mypy strict type checking
	@echo "──▶ Type-checking with mypy --strict..."
	uv run mypy orion/ --strict

# ── Podman ───────────────────────────────────────────────
podman-build: ## Build the Podman image
	@echo "──▶ Building Podman image..."
	podman build -t orion:latest .

podman-up: ## Start full stack via Podman Compose
	@echo "──▶ Starting Podman Compose stack..."
	podman-compose -f podman-compose.yml up -d

podman-up-dev: ## Start stack in dev mode (hot reload)
	@echo "──▶ Starting Podman Compose dev stack..."
	podman-compose -f podman-compose.yml --profile dev up -d

podman-down: ## Stop Podman Compose stack
	@echo "──▶ Stopping Podman Compose stack..."
	podman-compose -f podman-compose.yml down

# ── Operations ───────────────────────────────────────────
eval: ## Run evaluation suite against sample tasks
	@echo "──▶ Running eval suite..."
	PYTHONUTF8=1 uv run python scripts/eval_task.py $(ARGS)

health: ## Run system health check
	@echo "──▶ Running health check..."
	uv run python scripts/healthcheck.py

seed: ## Seed the tool registry
	@echo "──▶ Seeding tool registry..."
	uv run python scripts/seed_registry.py

# ── Cleanup ──────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	@echo "──▶ Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf eval_reports/ logs/
	@echo "  ✓ Clean complete"
