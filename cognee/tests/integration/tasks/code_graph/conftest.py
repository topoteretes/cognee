"""Real local stores and Git remotes for the public code ingestion path."""

import importlib
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest_asyncio

from cognee.base_config import get_base_config
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
from cognee.infrastructure.databases.relational import (
    get_migration_config,
    get_relational_config,
    get_relational_engine,
)
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.tasks.code_graph.enola import find_enola_binary


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class Remote:
    def __init__(self, root, name, git_config):
        self.url = f"https://github.com/{name}"
        self.key = f"github.com/{name}"
        self.work = root / name
        self.work.mkdir(parents=True)
        git(self.work, "init", "-q", "-b", "main")
        git(self.work, "config", "uploadpack.allowFilter", "true")
        self.commit("original")
        # Git rewrites the transport, while origin.url keeps its hosted URL.
        git(
            root,
            "config",
            "--file",
            str(git_config),
            f"url.{self.work.as_uri()}.insteadOf",
            self.url,
        )

    def commit(self, symbol):
        (self.work / "main.py").write_text(
            f"def {symbol}():\n    return shared()\n\ndef shared():\n    return 1\n"
        )
        git(self.work, "add", "main.py")
        git(self.work, "commit", "-q", "-m", symbol)
        return git(self.work, "rev-parse", "HEAD")


@pytest_asyncio.fixture
async def code_environment(tmp_path, monkeypatch):
    # Enola is a core dependency. A missing binary is a broken installation,
    # so this suite must fail rather than silently disappear from CI.
    find_enola_binary()
    for key, value in {
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
        "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
        "VECTOR_DB_SUBPROCESS_ENABLED": "false",
        "COGNEE_REPOS_DIR": str(tmp_path / "clones"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Code ingestion test",
        "GIT_AUTHOR_EMAIL": "code@example.test",
        "GIT_COMMITTER_NAME": "Code ingestion test",
        "GIT_COMMITTER_EMAIL": "code@example.test",
    }.items():
        monkeypatch.setenv(key, value)
    for name in list(os.environ):
        if name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            monkeypatch.delenv(name)
    for name in (
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_TEMPLATE_DIR",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    git_config = tmp_path / "gitconfig"
    git_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_config))
    git(tmp_path, "config", "--file", str(git_config), "protocol.file.allow", "always")

    configurations = (
        (
            get_base_config(),
            {
                "system_root_directory": str(tmp_path / "system"),
                "data_root_directory": str(tmp_path / "data"),
                "repos_root_directory": str(tmp_path / "clones"),
            },
        ),
        (get_relational_config(), {"db_provider": "sqlite", "db_path": str(tmp_path / "db")}),
        (
            get_migration_config(),
            {"migration_db_provider": "sqlite", "migration_db_path": str(tmp_path / "db")},
        ),
        (
            get_graph_config(),
            {
                "graph_database_provider": "ladybug",
                "graph_dataset_database_handler": "ladybug",
                "graph_file_path": str(tmp_path / "graph"),
                "graph_database_subprocess_enabled": False,
            },
        ),
        (
            get_vectordb_config(),
            {
                "vector_db_provider": "lancedb",
                "vector_dataset_database_handler": "lancedb",
                "vector_db_url": str(tmp_path / "vectors"),
                "vector_db_subprocess_enabled": False,
            },
        ),
    )
    for config, values in configurations:
        for key, value in values.items():
            monkeypatch.setattr(config, key, value)

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)
    # DNS is the only remote service replaced. Git, extraction and storage run.
    ssrf = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")
    monkeypatch.setattr(ssrf, "validate_outbound_url", AsyncMock())
    assert str(tmp_path) in str(get_relational_engine().engine.url)
    await setup()
    user = await get_default_user()

    yield SimpleNamespace(
        user=user,
        root=tmp_path,
        remote=lambda name: Remote(tmp_path, name, git_config),
    )

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    await get_relational_engine().engine.dispose()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)
