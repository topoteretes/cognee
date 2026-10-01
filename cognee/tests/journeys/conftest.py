"""Journey test fixtures.

Two execution modes, chosen by ``COGNEE_JOURNEY_MODE``:

* ``mock`` (default): deterministic LLM and embeddings from ``mock_ai``. No
  network, no secrets, byte-for-byte reproducible. This is the tier every PR
  runs, including fork PRs.
* ``llm``: real providers from the environment. Correctness journeys switch to
  threshold assertions and additionally enforce ``forbidden`` tokens.

Environment is pinned before ``cognee`` is imported so config caches see it.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

# --- environment must be set before importing cognee -------------------------
_ENV_PINS = {
    "TELEMETRY_DISABLED": "1",
    "ENV": "dev",
    "REQUIRE_AUTHENTICATION": "true",
    "ENABLE_BACKEND_ACCESS_CONTROL": "true",
    "HASH_API_KEY": "false",
    "COGNEE_SKIP_CONNECTION_TEST": "true",
    "RUNTIME__LOG_LEVEL": "ERROR",
}
for _key, _value in _ENV_PINS.items():
    os.environ.setdefault(_key, _value)

from cognee.tests.journeys import _support, mock_ai  # noqa: E402

_MOCK_STATE: dict = {"llm": None}


@pytest.fixture(scope="session", autouse=True)
def _install_ai_mocks():
    """Swap LLM and embeddings for deterministic stand-ins in mock mode.

    A fixture rather than import-time code so collecting this directory alongside
    other suites does not patch anything until a journey actually runs; the
    patches are removed again when the session ends so later suites in the same
    process see the real gateway.
    """
    if _support.IS_MOCK and _MOCK_STATE["llm"] is None:
        # Keys are never used, but config validation wants them present.
        with patch("dotenv.load_dotenv"):
            _MOCK_STATE["llm"] = mock_ai.install_all(
                _support.mock_graphs(_support.load_documents())
            )
    yield _MOCK_STATE["llm"]
    if _MOCK_STATE["llm"] is not None:
        mock_ai.uninstall_all()
        _MOCK_STATE["llm"] = None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "quickstart: builds a wheel and installs it in a fresh venv (slow)"
    )
    config.addinivalue_line("markers", "journey: high-level product contract test")


@pytest.fixture(scope="session")
def journey_mode() -> str:
    return _support.MODE


@pytest.fixture(scope="session")
def mock_llm(_install_ai_mocks):
    """The installed MockLLM in mock mode, else None."""
    return _install_ai_mocks


@pytest.fixture(scope="session")
def corpus() -> list[_support.Document]:
    return _support.load_documents()


@pytest.fixture(scope="session")
def questions() -> list[_support.Question]:
    return _support.load_questions()


def _clear_engine_caches() -> None:
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()


async def _reset_engines_and_prune() -> None:
    import cognee

    _clear_engine_caches()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest_asyncio.fixture
async def clean_env(tmp_path):
    """Fresh data + system roots under tmp_path, pruned before and after.

    The previous roots are restored on teardown so the journey does not leave
    global config pointing into a temporary directory for later tests.
    """
    import cognee
    from cognee.base_config import get_base_config
    from cognee.modules.engine.operations.setup import setup as engine_setup

    base_config = get_base_config()
    previous_roots = (base_config.data_root_directory, base_config.system_root_directory)

    root = Path(tmp_path)
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))
    await _reset_engines_and_prune()
    await engine_setup()
    try:
        yield root
    finally:
        await _reset_engines_and_prune()
        cognee.config.data_root_directory(previous_roots[0])
        cognee.config.system_root_directory(previous_roots[1])
        _clear_engine_caches()


@pytest_asyncio.fixture
async def default_user(clean_env):
    from cognee.modules.users.methods import get_default_user

    return await get_default_user()
