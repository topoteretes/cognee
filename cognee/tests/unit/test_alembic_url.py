"""The live engine URL must survive Alembic's ConfigParser interpolation."""

import runpy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from alembic import context
from alembic.config import Config
from sqlalchemy.engine import URL, make_url

PACKAGE = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "url",
    [
        URL.create("postgresql+asyncpg", username="user@example.com", password="secret", host="db"),
        URL.create("postgresql+asyncpg", username="user", password="p@ss:/%", host="db"),
        URL.create("postgresql+asyncpg", username="user%40example.com", host="db"),
        URL.create("postgresql+asyncpg", username="user", password="secret", host="db"),
        URL.create("sqlite+aiosqlite", database="cache%20directory/database.db"),
    ],
    ids=["email-user", "reserved-password", "literal-percent", "plain-url", "sqlite-percent"],
)
def test_alembic_env_preserves_live_engine_url(url):
    config = Config(str(PACKAGE / "alembic.ini"))
    config.attributes["configure_logger"] = False
    config.attributes["relational_engine"] = SimpleNamespace(engine=SimpleNamespace(url=url))

    with (
        patch.object(context, "config", config, create=True),
        patch.object(context, "is_offline_mode", return_value=True),
        patch.object(context, "configure") as configure,
        patch.object(context, "begin_transaction", side_effect=nullcontext),
        patch.object(context, "run_migrations"),
    ):
        runpy.run_path(str(PACKAGE / "alembic" / "env.py"))

    configured_url = configure.call_args.kwargs["url"]
    assert configured_url == url.render_as_string(hide_password=False)
    assert make_url(configured_url) == url
