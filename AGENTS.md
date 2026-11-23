# Repository guidance

This file provides shared project instructions for coding agents. `CLAUDE.md` imports it. Update this file when a rule applies across the repository. Put feature behavior, configuration tables, debugging notes, and worked examples in the relevant topic docs or source docstrings. Link to references an agent should consult for specific tasks.

## Design Principles

1. **Don't overengineer**: Simple beats complex.
2. **No fallbacks**: One correct path, no alternatives.
3. **One way**: One way to do things, not many.
4. **Throw errors**: Fail fast when the operation's preconditions aren't met. Observability (telemetry, logging) is best-effort and never changes the outcome of the operation.
5. **No backups**: Trust the primary mechanism.
6. **Separation of concerns**: Each function has a single responsibility.
7. **Be concise but informative**: No essays, but the docstrings and comments a reader needs.
8. **No invented limits**: Never add a cutoff (`max_n`, `[:k]`, "older than X") or an expiry (TTL, cooldown, eviction) the task did not ask for — both are silent data loss. If you believe one is unavoidable, stop and ask, stating what would be dropped and why; never add it first and justify it afterwards. Existing bounds (lock leases, recovery floors, batch caps) are deliberate: removing or changing one is the same question.
9. **Honor DRY principle**: Before adding a helper, utility or wrapper, search for an existing one in the package's utils and cognee/shared/. Extend what exists rather than adding a near-duplicate, unless it conflicts with principle #1. If the existing helper doesn't fit, say why in the PR.

**Methodology**: surgical changes only; evidence-based debugging (minimal, targeted logging); fix root causes, not symptoms. Debug like a detective: form the theory of the crime, collect evidence, and change code only once the evidence proves it.

## Working in this repository

- Branch from `dev`. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for signed-off commits and PR titles. Core-team PRs must include a Linear issue key in the branch name or PR title.
- Keep Python names in `snake_case` and class names in `PascalCase`. Use type annotations for public APIs where practical and check the generated Swagger schema for API signature changes. Use shared logging helpers in `cognee.shared.logging_utils`.
- Add feature-specific code to the owning package. Prefer extending the system with tasks, loaders, or retrievers when the existing extension points fit.
- Place tests under [`cognee/tests/`](cognee/tests/README.md). Use `test_*.py` names and `pytest.mark.asyncio` for async tests. Use fixtures and CI-provided settings for tests that need external services.
- When a change adds a feature users would build on (a new operation, mode or extension point), add an example and list it in [examples/README.md](examples/README.md). If an existing example already covers that feature, update it instead. A new parameter or setting on an existing feature doesn't need an example.
- Choose the example folder using the [example contribution guide](examples/README.md#-contributing-a-new-example). Use `examples/guides/` for one self-contained lesson, `examples/advanced_guides/` for a deeper treatment of a topic with a basic guide, `examples/demos/` for a scenario in a topic subfolder, or `examples/cookbooks/` for a reusable application.
- Edit agent skills under `.agents/skills/`. The `.claude/skills` symlink points there. On Windows the [`SessionStart` hook](.claude/hooks/link-skills.sh) creates a junction and marks the path skip-worktree. If working on the link itself on a Windows clone, see the instructions in that hook.

## Repository layout

- [`cognee/`](cognee/) contains the Python SDK, HTTP API (`api/`), CLI (`cli/`), domain logic (`modules/`), pipeline tasks (`tasks/`), database, LLM, and loader adapters (`infrastructure/`), shared helpers (`shared/`), relational migrations (`alembic/`), and tests (`tests/`). Graph and vector data migrations live in [`cognee/modules/migrations/`](cognee/modules/migrations/).
- [`cognee_db_workers/`](cognee_db_workers/) contains subprocess workers for the embedded graph and vector engines.
- [`cognee-mcp/`](cognee-mcp/) contains the MCP server. [`cognee-frontend/`](cognee-frontend/) contains the Next.js UI.
- [`cognee-integrations/`](cognee-integrations/) contains agent plugins. [`cognee-connectors/`](cognee-connectors/) contains connector packages.
- [`examples/`](examples/) and [`notebooks/`](notebooks/) contain runnable examples. See the [example contribution guide](examples/README.md#-contributing-a-new-example) for the folder rules.
- [`docs/`](docs/) contains contributor and agent reference material.
- [`tools/`](tools/) contains CI and automation checks.
- [`deployment/`](deployment/) contains Helm charts. [`distributed/deploy/`](distributed/deploy/) contains deployment templates.
- [`catalog/`](catalog/) contains YAML cards for the Integrations Hub and Use-Case Gallery. CI validates the cards. Update a card when you move a file it links to.
- [`.agents/skills/`](.agents/skills/) contains agent skills. `.claude/skills` links to this directory.

## Common commands

Python 3.10–3.14 is supported. Run these from the repository root.

```bash
uv sync --dev
uv run pytest cognee/tests/unit/
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```

Use the [test guide](cognee/tests/README.md) to choose integration and end-to-end suites. For local setup and launch commands, see [CONTRIBUTING.md](CONTRIBUTING.md), the [MCP server guide](cognee-mcp/README.md), and the [frontend guide](cognee-frontend/README.md).

## Technical references

- [Architecture](docs/architecture.md) for component boundaries and dataset isolation.
- [Memory internals](docs/memory-internals.md) for memory operations, extension points, and feature behavior.
- [Runtime configuration](docs/runtime-configuration.md) and [`.env.template`](.env.template) for provider setup, performance flags, security settings, and troubleshooting.
- [Retrieval internals](docs/retrieval-internals.md) and [`recall()` vs `search()`](docs/recall-vs-search.md) for search types, routing, session memory, and completion prompts.
