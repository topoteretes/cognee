"""code_docs at the cognify() boundary: resolved once, bound on both code routes' task lists."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
config_module = importlib.import_module("cognee.modules.cognify.config")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


@pytest.fixture(autouse=True)
def fresh_cognify_config(monkeypatch):
    """CODE_GRAPH_DOCS as the test sets it: get_cognify_config caches across the process."""
    monkeypatch.delenv("CODE_GRAPH_DOCS", raising=False)
    config_module.get_cognify_config.cache_clear()
    yield
    config_module.get_cognify_config.cache_clear()


async def _code_route_builders(**cognify_kwargs):
    """Run cognify() with the pipeline stubbed out; return the two code route builders' mocks.

    The extractor is pinned to "llm" so a keyless environment never reaches the
    GLiNER runtime install; the route builders are what this test looks at.
    """
    with (
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(cognify_module, "resolve_extractor", return_value="llm"),
        patch.object(cognify_module, "ensure_extractor_runtime", new=AsyncMock()),
        patch.object(
            cognify_module, "get_pipeline_executor", return_value=AsyncMock(return_value={})
        ),
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value=[])),
        patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value=[])),
        patch.object(
            cognify_module, "get_code_file_tasks", new=MagicMock(return_value=[])
        ) as files,
        patch.object(
            cognify_module, "get_code_repo_tasks", new=MagicMock(return_value=[])
        ) as repos,
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new=AsyncMock()),
        patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None),
    ):
        await cognify_module.cognify(**cognify_kwargs)
    return files, repos


@pytest.mark.asyncio
async def test_cognify_binds_the_setting_on_both_code_routes():
    files, repos = await _code_route_builders()

    files.assert_called_once_with(code_docs=True)
    repos.assert_called_once_with(code_docs=True)


@pytest.mark.asyncio
async def test_the_argument_wins_over_the_setting(monkeypatch):
    files, repos = await _code_route_builders(code_docs=False)
    files.assert_called_once_with(code_docs=False)
    repos.assert_called_once_with(code_docs=False)

    monkeypatch.setenv("CODE_GRAPH_DOCS", "false")
    config_module.get_cognify_config.cache_clear()
    files, repos = await _code_route_builders()
    files.assert_called_once_with(code_docs=False)
    repos.assert_called_once_with(code_docs=False)
    files, repos = await _code_route_builders(code_docs=True)
    files.assert_called_once_with(code_docs=True)
    repos.assert_called_once_with(code_docs=True)


@pytest.mark.asyncio
async def test_cognify_rejects_a_non_bool_or_remote_code_docs(monkeypatch):
    with pytest.raises(ValueError, match="code_docs"):
        await cognify_module.cognify(code_docs="false")

    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: MagicMock())
    with pytest.raises(ValueError, match="remote Cognee instance"):
        await cognify_module.cognify(code_docs=False)
