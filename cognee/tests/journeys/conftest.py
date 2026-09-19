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

from cognee.tests.journeys import _support  # noqa: E402
from cognee.tests.journeys import mock_ai  # noqa: E402

_MOCK_STATE: dict = {"llm": None}


@pytest.fixture(scope="session", autouse=True)
def _install_ai_mocks():
    """Swap LLM and embeddings for deterministic stand-ins in mock mode.

    A fixture rather than import-time code so collecting this directory alongside
    other suites does not patch anything until a journey actually runs.
    """
    if _support.IS_MOCK and _MOCK_STATE["llm"] is None:
        # Keys are never used, but config validation wants them present.
        with patch("dotenv.load_dotenv"):
            _MOCK_STATE["llm"] = mock_ai.install_all(
                _support.mock_graphs(_support.load_documents())
            )
    yield _MOCK_STATE["llm"]


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


async def _reset_engines_and_prune() -> None:
    import cognee
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest_asyncio.fixture
async def clean_env(tmp_path):
    """Fresh data + system roots under tmp_path, pruned before and after."""
    import cognee
    from cognee.modules.engine.operations.setup import setup as engine_setup

    root = Path(tmp_path)
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))
    await _reset_engines_and_prune()
    await engine_setup()
    try:
        yield root
    finally:
        await _reset_engines_and_prune()


@pytest_asyncio.fixture
async def default_user(clean_env):
    from cognee.modules.users.methods import get_default_user

    return await get_default_user()
