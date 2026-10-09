# cognee/tests — layout, how to run, what needs credentials

## Layout

| Folder | What lives there | Needs |
|---|---|---|
| `unit/` | Fast, mocked tests mirroring the source tree (`unit/modules/retrieval/`, `unit/infrastructure/...`, `unit/tasks/...`). Session-scoped `unit/conftest.py` creates the SQLite relational DB once | nothing external |
| `integration/` | Real local databases (Ladybug + LanceDB by default), real pipelines; most call an LLM | `LLM_API_KEY` (+ embedding provider) |
| `e2e/` | Full-stack suites run per backend in CI: `incremental_update/`, `postgres/`, `docker_compose/`, `migrations/`, `dataset_queue/` | the backend named in the folder (Postgres, Neo4j, Docker, …) |
| `api/` | FastAPI endpoint tests (agents, agent mode) | `LLM_API_KEY` for some |
| `cli_tests/` | `cli_unit_tests/` (argparse, mocked) and `cli_integration_tests/` (real commands) | integration half needs `LLM_API_KEY` |
| `deployment/` | Deploy-target smoke tests, marked `@pytest.mark.deployment` | a deployed instance |
| `migrations/`, `release_migration/`, `backwards_compatibility/` | Alembic and data-format migration checks against frozen fixtures | nothing external |
| `performance/`, `subprocesses/` | Benchmarks and subprocess-engine stress scripts; mostly run by hand or dedicated workflows | varies |
| `tasks/` | Task-level tests that use `*_test.py` naming (translation) | `LLM_API_KEY` |
| `test_data/` | Fixtures: documents, the code-repo fixture, frozen schemas | — |
| `test_*.py` (top level, 72 files) | Scenario tests, most run by their own CI job (16 are referenced by no workflow -- the neptune, remote-kuzu, loader and a few permission/pipeline files): delete semantics (`test_delete_*`), permissions, search DB matrix (`test_search_db.py`), custom models, code graph, usage logging, … | most need `LLM_API_KEY`; the backend ones need that backend |

Test files are named `test_*.py` or `*_test.py` (both are collected). Async tests are
marked `@pytest.mark.asyncio` (pytest-asyncio in strict mode).

## Running

```bash
uv run pytest cognee/tests/unit/               # what every PR runs; no credentials needed
uv run pytest                                  # = cognee/tests (testpaths in pyproject); see the note below
uv run pytest cognee/tests/integration/        # needs LLM_API_KEY
uv run pytest cognee/tests/unit/ --splits 3 --group 1   # CI shards with pytest-split
uv run pytest cognee/tests/test_search_db.py -v --log-level=INFO   # one scenario file
```

A bare `uv run pytest` does not currently finish: six modules fail at collection and
pytest aborts the run. All six predate `testpaths` and none is reached by CI, which
always passes an explicit path. Two neptune scenario tests build a live adapter at
import; `tasks/summarization/summarize_code_test.py` imports a fixture module that no
longer exists; and three files under `integration/` share a basename with a file under
`unit/`, which collides only when both trees are collected in one process (and hides
56 tests when it does). Until those are fixed, run a path.

Tests read `.env` through `cognee` itself, so a populated `.env` in the repo root is
enough locally. CI supplies the same variables from secrets; the workflows under
`.github/workflows/` are the reference for which backend each e2e suite expects
(`basic_tests.yml` is the per-PR gate).

## Running without API keys

- `unit/` and `cli_tests/cli_unit_tests/` never call a provider.
- Elsewhere, tests guard themselves with `pytest.mark.skipif`: on `LLM_API_KEY`
  being unset (`has_llm_api_key()` helpers), and on optional extras being importable
  (`HAS_LANCEDB`, `HAS_NEO4J`, `HAS_PGVECTOR`, `HAS_LADYBUG`, `HAS_TURSO`). A missing
  key or extra skips, it does not fail.
- `MOCK_EMBEDDING=true` and `MOCK_CODE_SUMMARY=true` replace the embedding and code
  summary calls with deterministic stand-ins in the tests that support them.
- `extractor="gliner_demo"` pipelines and `SearchType.CHUNKS` recall run with no LLM key
  at all; see `examples/guides/no_llm_remember_recall.py`.

## Conventions

- Mirror the source path: a test for `cognee/modules/x/y.py` goes in
  `cognee/tests/unit/modules/x/`.
- New public API → an integration test covering `remember → recall` (or
  `add → cognify → search` when the feature lives in one stage), plus an example in
  `examples/guides/` listed in `examples/README.md`.
- Local databases are written under `.cognee_system/` and `.data_storage/`; tests that
  need a clean slate call `cognee.prune.prune_data()` / `prune_system(metadata=True)`.
- `test_subprocess_rss.py` is a benchmark script, not a test; the root `conftest.py`
  excludes it from collection.
