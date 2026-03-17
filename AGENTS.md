## Repository Guidelines

This document summarizes how to work with the cognee repository: how it’s organized, how to build, test, lint, and contribute. It mirrors our actual tooling and CI while providing quick commands for local development.

## Project Structure & Module Organization

- `cognee/`: Core Python library and API.
  - `api/`: FastAPI application and versioned routers (add, cognify, memify, search, delete, users, datasets, responses, visualize, settings, sync, update, checks).
  - `cli/`: CLI entry points and subcommands invoked via `cognee` / `cognee-cli`.
  - `infrastructure/`: Databases, LLM providers, embeddings, loaders, and storage adapters.
  - `modules/`: Domain logic (graph, retrieval, ontology, users, processing, observability, etc.).
  - `tasks/`: Reusable tasks (e.g., code graph, web scraping, storage). Extend with new tasks here.
  - `eval_framework/`: Evaluation utilities and adapters.
  - `shared/`: Cross-cutting helpers (logging, settings, utils).
  - `tests/`: Unit, integration, CLI, and end-to-end tests organized by feature.
  - `__main__.py`: Entrypoint to route to CLI.
- `cognee-mcp/`: Model Context Protocol server exposing cognee as MCP tools (SSE/HTTP/stdio). Contains its own README and Dockerfile.
- `cognee-frontend/`: Next.js UI for local development and demos.
- `distributed/`: Utilities for distributed execution (Modal, workers, queues).
- `examples/`: Example scripts demonstrating the public APIs and features (graph, code graph, multimodal, permissions, etc.).
- `notebooks/`: Jupyter notebooks for demos and tutorials.
- `alembic/`: Database migrations for relational backends.

Notes:
- Co-locate feature-specific helpers under their respective package (`modules/`, `infrastructure/`, or `tasks/`).
- Extend the system by adding new tasks, loaders, or retrievers rather than modifying core pipeline mechanisms.

## Build, Test, and Development Commands

Python (root) – requires Python >= 3.10 and < 3.14. We recommend `uv` for speed and reproducibility.

- Create/refresh env and install dev deps:
```bash
uv sync --dev --all-extras --reinstall
```

- Run the CLI (examples):
```bash
uv run cognee-cli add "Cognee turns documents into AI memory."
uv run cognee-cli cognify
uv run cognee-cli search "What does cognee do?"
uv run cognee-cli -ui   # Launches UI, backend API, and MCP server together
```

- Start the FastAPI server directly:
```bash
uv run python -m cognee.api.client
```

- Run tests (CI mirrors these commands):
```bash
uv run pytest cognee/tests/unit/ -v
uv run pytest cognee/tests/integration/ -v
```

- Lint and format (ruff):
```bash
uv run ruff check .
uv run ruff format .
```

- Optional static type checks (mypy):
```bash
uv run mypy cognee/
```

MCP Server (`cognee-mcp/`):

- Install and run locally:
```bash
cd cognee-mcp
uv sync --dev --all-extras --reinstall
uv run python src/server.py               # stdio (default)
uv run python src/server.py --transport sse
uv run python src/server.py --transport http --host 127.0.0.1 --port 8000 --path /mcp
```

- API Mode (connect to a running Cognee API):
```bash
uv run python src/server.py --transport sse --api-url http://localhost:8000 --api-token YOUR_TOKEN
```

- Docker quickstart (examples): see `cognee-mcp/README.md` for full details
```bash
docker run -e TRANSPORT_MODE=http --env-file ./.env -p 8000:8000 --rm -it cognee/cognee-mcp:main
```

Frontend (`cognee-frontend/`):
```bash
cd cognee-frontend
npm install
npm run dev     # Next.js dev server
npm run lint    # ESLint
npm run build && npm start
```

## Coding Style & Naming Conventions

Python:
- 4-space indentation, modules and functions in `snake_case`, classes in `PascalCase`.
- Public APIs should be type-annotated where practical.
- Use `ruff format` before committing; `ruff check` enforces import hygiene and style (line-length 100 configured in `pyproject.toml`).
- Prefer explicit, structured error handling. Use shared logging utilities in `cognee.shared.logging_utils`.

MCP server and Frontend:
- Follow the local `README.md` and ESLint/TypeScript configuration in `cognee-frontend/`.

## Testing Guidelines

- Place Python tests under `cognee/tests/`.
  - Unit tests: `cognee/tests/unit/`
  - Integration tests: `cognee/tests/integration/`
  - CLI tests: `cognee/tests/cli_tests/`
- Name test files `test_*.py`. Use `pytest.mark.asyncio` for async tests.
- Avoid external state; rely on test fixtures and the CI-provided env vars when LLM/embedding providers are required. See CI workflows under `.github/workflows/` for expected environment variables.
- When adding public APIs, provide/update targeted examples under `examples/python/`.

## Commit & Pull Request Guidelines

- Use clear, imperative subjects (≤ 72 chars) and conventional commit styling in PR titles. Our CI validates semantic PR titles (see `.github/workflows/pr_lint`). Examples:
  - `feat(graph): add temporal edge weighting`
  - `fix(api): handle missing auth cookie`
  - `docs: update installation instructions`
- Reference related issues/discussions in the PR body and provide brief context.
- PRs should describe scope, list local test commands run, and mention any impacts on MCP server or UI if applicable.
- Sign commits and affirm the DCO (see `CONTRIBUTING.md`).

## CI Mirrors Local Commands

Our GitHub Actions run the same ruff checks and pytest suites shown above (`.github/workflows/basic_tests.yml` and related workflows). Use the commands in this document locally to minimize CI surprises.
