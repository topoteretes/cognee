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
10. **Keep the event loop responsive**: Use async I/O and offload blocking I/O or CPU-heavy work from async request paths, especially stages that may take a second or more; verify other requests can complete while they run.

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

## Architecture checks (enola)

[enola](https://github.com/enola-labs/enola) grades what a change does to the structure of the code: layers crossed the wrong way, new cycles, new coupling. The intended architecture is declared in [`enola-intent.yaml`](enola-intent.yaml): the layer order from [docs/architecture.md](docs/architecture.md#layer-structure) and the code paths the docs describe as making no LLM call. The `Architecture (enola)` CI job reports what each pull request does to it. It is report-only and fails only when `enola-intent.yaml` is invalid or one of its layers or components matches nothing.

If you do not already use enola while developing, run it locally:

- enola comes with cognee: `uv sync --dev` installs the `enola-cli` that `pyproject.toml` pins, the version CI uses. Run the commands below as `uv run enola …`.
- Before changing shared code, see what depends on it: `enola impact <symbol>`, or `enola plan --paths <files>` for the declared rules that govern those files.
- Before opening a pull request, grade the branch: `enola check --base origin/dev`. It writes nothing. In a checkout whose directory is not named `cognee` (a `git worktree`, for example), it declines to compare (exit 3); there, run `enola baseline pin` before you start editing and `enola check` when done.
- With a coding agent, serve the graph over MCP: `claude mcp add enola -- <repo>/.venv/bin/enola <repo>`, where `<repo>` is the path to this repository.
- After editing `enola-intent.yaml`, `enola --generate . && enola constraints lint --fail-empty .` must exit 0. Every rule needs a `because:` that says why it exists.

`.enola/` holds local snapshots and is gitignored.

## Technical references

- [Architecture](docs/architecture.md) for component boundaries and dataset isolation.
- [Memory internals](docs/memory-internals.md) for memory operations, extension points, and feature behavior.
- [Runtime configuration](docs/runtime-configuration.md) and [`.env.template`](.env.template) for provider setup, performance flags, security settings, and troubleshooting.
- [Retrieval internals](docs/retrieval-internals.md) and [`recall()` vs `search()`](docs/recall-vs-search.md) for search types, routing, session memory, and completion prompts.
