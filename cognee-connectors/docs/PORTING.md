# Porting a provider from cognee core

This is the procedure used to bring every provider from cognee `dev` into this repo.
It is also the reference for a future port. Read `docs/ARCHITECTURE.md` first.

## Sources of truth (read-only)

- cognee `dev`: the checkout named in the port instructions. Never modify it.
- The cloud framework (`cloud-backend/cognee-saas-backend/modules/integrations/framework/`)
  is the reference the core contract was cut from.

## File mapping per provider

| From (cognee `dev`) | To (this repo) |
| --- | --- |
| `cognee/tasks/ingestion/connectors/<p>.py` | `connectors/<p>/cognee_connector_<p>/source.py` |
| `cognee/tasks/ingestion/connectors/<p>.md` | `connectors/<p>/README.md` (the base of it) |
| `cognee/modules/integrations/<p>/<p>_settings.py` | `.../settings.py` |
| `cognee/modules/integrations/<p>/client.py` | `.../client.py` |
| `cognee/modules/integrations/<p>/verify_<p>_signature.py` | `.../webhooks.py` on the core `WebhookVerifier`; the cognee shim stays in `cognee_host/` |
| `cognee/modules/integrations/<p>/adapter.py`, `sync.py`, `handle_*_event.py`, `agent_session.py`, `resume_sync.py`, `history*.py`, `home.py`, `persistence.py`, `oauth.py`, `channels.py`, `remember_message.py`, `response_url.py`, `handle_cognee_*.py`, `handle_slack_*.py` | `.../cognee_host/<same name>.py` |
| `cognee/modules/integrations/<p>/__init__.py` (registration side effect) | `.../cognee_host/__init__.py`: an explicit `register()`; no side effect at import |
| `cognee/api/v1/slack/routers/*.py` | `connectors/slack/cognee_connector_slack/cognee_host/routers.py` with `routers()` |
| `cognee/tests/unit/modules/integrations/test_<p>*.py`, `test_handle_<p>_*.py` and the provider's other unit tests | `connectors/<p>/tests/unit/` |
| `cognee/tests/unit/modules/integrations/conftest.py` | `connectors/<p>/tests/unit/conftest.py` (a copy, when a moved test uses `credential_db`) |
| `cognee/tests/integration/tasks/connectors/test_<p>_*.py`, `conftest.py`, `fake_linear.py` (Linear) | `connectors/<p>/tests/integration/` |
| `cognee/tests/integration/modules/integrations/test_slack_history_native.py` | `connectors/slack/tests/integration/` |
| `.env.template`, the provider's block | the README's Configuration section; names unchanged |
| `examples/guides/<p>.py`, `examples/python/integrations/*` | `connectors/<p>/examples/` with imports updated |
| `docs/slack-history.md`, `docs/integration-sync-progress.md` | `connectors/<p>/docs/` |

Pure modules that a provider shares with no host (a client, response-shape helpers,
`history_models.py`, `home.py` block builders, `response_url.py`) go next to `source.py`,
not under `cognee_host/`, when they import nothing from cognee.

## Import rewrites

| Was | Becomes |
| --- | --- |
| `from cognee.shared.logging_utils import get_logger` / `get_logger("x")` | `import logging` / `logging.getLogger(__name__)` |
| `from cognee.tasks.ingestion import dlt_utils` | `from cognee_connectors_core import dlt_conventions as dlt_utils` |
| `from cognee.tasks.ingestion.dlt_utils import X` | `from cognee_connectors_core.dlt_conventions import X` |
| `from cognee.modules.integrations.base import IntegrationSettings` | `from cognee_connectors_core.settings import ConnectorSettings` |
| `from cognee.modules.integrations.<p>.<m> import ...` | `from cognee_connector_<p>.<m> import ...` or `from cognee_connector_<p>.cognee_host.<m> import ...` |
| `from cognee.tasks.ingestion.connectors import <p>_source` | `from cognee_connector_<p> import <p>_source` |
| `from cognee.tasks.ingestion.connectors.<p> import ...` | `from cognee_connector_<p>.source import ...` |
| `from cognee.modules.integrations.google import ...` | `from cognee_connector_google import ...` |
| `from cognee.modules.integrations.oauth_flow import ...` | `from cognee_connectors_core.oauth_state import ...` |
| `if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < N: raise RuntimeError(...)` (the capability gate at the top of a source factory) | `dlt_conventions.require_document_sync_version(N, provider="<Provider>", feature="<what the host must read>")`, which reads the installed cognee's version; the package's own constant would compare the package with itself. A test that monkeypatched `module.dlt_utils.DOCUMENT_SYNC_VERSION` monkeypatches `cognee_connectors_core.dlt_conventions.host_document_sync_version` instead. |
| `cognee.modules.integrations.{base,credentials,registry,crypto,models,connect,sync_progress,ingestion}` | unchanged: these are the host seam; allowed only under `cognee_host/` |

Docstring examples that import from cognee are rewritten to the package import.
`import cognee` in an example stays: the example runs with cognee installed.

Code that needs cognee at call time for a host feature (Google Drive's file text
extraction through cognee's loaders) keeps a function-level `import cognee...` with a
clear `ImportError` message naming `pip install cognee-connector-<p>[cognee]`. The
module still imports without cognee.

## The contract object

`connector.py` defines `CONNECTOR`, the provider on the shared contract:

- Tier 1 (standard OAuth2, account id in the token response): a `ConnectorDescriptor`
  wrapped by `DescriptorConnector`. Endpoints, scopes and the dotted paths come from
  the provider's `adapter.py` in cognee and from the cloud connector of the same
  provider when one exists.
- Tier 2 (identity needs one extra API call): a `DescriptorConnector` subclass that
  overrides `parse_installation` (or the hook core provides for it).
- Tier 3 (not OAuth2 with a bearer token; GitHub Apps): a `Connector` subclass.
- API-key and local providers (Apollo, Asana, OneNote, Obsidian): a `ConnectorDescriptor` with `auth` set to `api_key` or `local`; the OAuth members are unused and the descriptor still names the provider, its dataset naming and its sources for hosts and the catalog.

The contract object is tested with the provider's real token-response shape, which the
moved unit tests already carry as fixtures, through `cognee_connectors_core.testing`'s
`FakeHttpClient`. Nothing in `connector.py` imports cognee.

## `cognee_host/`

The moved host-bound modules keep their logic. `cognee_host/__init__.py` exposes:

```python
def register() -> None:
    """Register this provider with cognee's integration registry (cognee >= 1.6.5)."""
    from cognee.modules.integrations.registry import use_integration
    from .adapter import <Provider>Integration
    use_integration(<Provider>Integration())


def routers() -> list[tuple["APIRouter", str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags)."""
    return []
```

`register()` replaces the import-time `use_integration(...)` call the cognee package had.
Tests that relied on the side effect call `register()` in a fixture.

## pyproject template

```toml
[build-system]
requires = ["hatchling==1.32.0"]
build-backend = "hatchling.build"

[project]
name = "cognee-connector-<p-dashed>"
version = "0.1.0"
description = "<one line: what it syncs into cognee memory>"
readme = "README.md"
requires-python = ">=3.10,<3.15"
license = "Apache-2.0"
license-files = ["LICENSE"]
authors = [{ name = "Cognee team" }]
keywords = ["cognee", "ai-memory", "connector", "<provider>"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "cognee-connectors-core[dlt]>=0.1,<1",
    # the provider's own libraries, with the ranges cognee's pyproject uses
]

[project.optional-dependencies]
cognee = ["cognee>=1.6.5,<2"]

[project.urls]
Homepage = "https://github.com/topoteretes/cognee/tree/main/cognee-connectors/connectors/<p>"
Repository = "https://github.com/topoteretes/cognee"
Issues = "https://github.com/topoteretes/cognee/issues"
Documentation = "https://docs.cognee.ai"

[project.entry-points."cognee.connectors"]
<provider> = "cognee_connector_<p>:CONNECTOR"

[project.entry-points."cognee.sources"]
<provider> = "cognee_connector_<p>:<p>_source"

[project.entry-points."cognee.integrations"]
<provider> = "cognee_connector_<p>.cognee_host:register"

[tool.hatch.build.targets.wheel]
packages = ["cognee_connector_<p>"]

[tool.hatch.build.targets.sdist]
include = ["cognee_connector_<p>", "tests", "README.md", "LICENSE", "pyproject.toml"]
```

Every package directory carries a copy of the repo `LICENSE`. Dependency ranges copy
cognee's `pyproject.toml` for the same library (`aiohttp>=3.13.5,<4.0.0`,
`httpx>=0.28.1`, `google-api-python-client>=2.100.0,<3`, and so on).

## README template

1. Title and one paragraph: what it syncs, how (DLT source, documents), incremental and forget-on-delete.
2. Install: `pip install cognee-connector-<p>` and the `[cognee]` extra.
3. Configuration: every environment variable the settings class reads, with the
   provider's block from cognee's `.env.template`, names unchanged.
4. Usage: the SDK example (`remember(<p>_source(...))`).
5. Running on a cognee server: `register()` and, if any, `routers()`.
6. What is synced, what is not, limits (from the original `.md` guide).
7. Development: how to run the tests.

## Running the tests locally

Tests run with the workspace's own interpreter, `.venv/bin/python`, created by `uv sync --group dev` at the repo root (cognee 1.6.5 from PyPI plus pytest and ruff). From the repo root:

```bash
export COGNEE_ENV_FILE=<an empty file>   # keep cognee from loading any .env
export TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/google:connectors/<p> python -m pytest connectors/<p>/tests -q -p no:cacheprovider -o addopts=""
```

Once the workspace is synced (`uv sync --all-packages --group dev`), `uv run pytest
connectors/<p>/tests` does the same.

## Definition of done for a ported provider

- Every file in the mapping exists in the package; nothing of the provider is left behind.
- `python scripts/check_cognee_free.py cognee_connector_<p>` passes (the package imports with cognee absent).
- `ruff check` and `ruff format --check` pass on the package.
- The moved tests pass unchanged apart from import paths and the explicit `register()` call.
- `connector.py` has tests against the provider's real token-response shape.
- The README documents every environment variable the settings class reads.
- `uv build --package cognee-connector-<p>` produces a wheel and an sdist, and the
  entry points resolve after `pip install` of the wheel.
