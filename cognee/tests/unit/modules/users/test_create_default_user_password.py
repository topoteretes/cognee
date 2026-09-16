"""The default superuser must not ship with a publicly known password (SDK-549).

Before this, ``create_default_user`` fell back to the literal
``default_password``, so every install carried a superuser whose credentials
were public. These tests pin the two halves of the replacement: an unset
``DEFAULT_USER_PASSWORD`` yields an unpredictable password that is never
recorded, and a set one is still honoured verbatim.
"""

import importlib
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.base_config import get_base_config
from cognee.modules.users.methods.create_default_user import (
    DEFAULT_USER_EMAIL,
    create_default_user,
)

# The ``methods`` package re-exports the function under its module's own name,
# so patching by dotted path resolves to the function. Hold the module itself.
create_default_user_module = importlib.import_module(
    "cognee.modules.users.methods.create_default_user"
)
get_default_user_module = importlib.import_module("cognee.modules.users.methods.get_default_user")


@pytest.fixture
def created_user_password(monkeypatch):
    """Run create_default_user with create_user mocked; return the password it was given."""

    async def _run(configured_password: str | None):
        if configured_password is None:
            monkeypatch.delenv("DEFAULT_USER_PASSWORD", raising=False)
        else:
            monkeypatch.setenv("DEFAULT_USER_PASSWORD", configured_password)
        # BaseConfig is lru_cached, so the env change only lands on a fresh read.
        get_base_config.cache_clear()
        with patch.object(
            create_default_user_module,
            "create_user",
            new=AsyncMock(return_value=object()),
        ) as create_user_mock:
            await create_default_user()
        get_base_config.cache_clear()
        return create_user_mock.call_args.kwargs

    return _run


@pytest.mark.asyncio
async def test_unset_password_is_random_and_not_the_known_literal(created_user_password):
    """No DEFAULT_USER_PASSWORD: the account exists but no known value opens it."""
    first = await created_user_password(None)
    second = await created_user_password(None)

    assert first["email"] == DEFAULT_USER_EMAIL
    assert first["is_superuser"] is True
    # The specific defect: the old fallback was this exact string.
    assert first["password"] != "default_password"
    # Random, not merely different: two creations must not agree.
    assert first["password"] != second["password"]
    assert len(first["password"]) >= 32


@pytest.mark.asyncio
async def test_configured_password_is_used_verbatim(created_user_password):
    """DEFAULT_USER_PASSWORD set: unchanged behaviour, so operators keep login."""
    kwargs = await created_user_password("operator-chosen-password")

    assert kwargs["password"] == "operator-chosen-password"
    assert kwargs["email"] == DEFAULT_USER_EMAIL
    assert kwargs["is_superuser"] is True


class TestExistingDeploymentsAreUnaffected:
    """Upgrading must not touch an already-created default user (SDK-549).

    The password change only runs at creation time. A deployment that already
    has the default-user row keeps whatever password that row was created with
    -- including the historical ``default_password`` -- because
    ``get_default_user`` returns the existing row and never reaches
    ``create_default_user``.
    """

    @staticmethod
    def _engine_returning(user) -> MagicMock:
        @asynccontextmanager
        async def session_cm():
            session = MagicMock()
            result = MagicMock()
            result.scalars.return_value.first.return_value = user
            session.execute = AsyncMock(return_value=result)
            yield session

        engine = MagicMock()
        engine.get_async_session = session_cm
        return engine

    @pytest.mark.asyncio
    async def test_existing_default_user_is_returned_without_recreation(self, monkeypatch):
        """The upgrade path: row present, so no creation and no new password."""
        monkeypatch.delenv("DEFAULT_USER_PASSWORD", raising=False)
        get_base_config.cache_clear()

        existing_user = object()
        create_mock = AsyncMock()
        with (
            patch.object(
                get_default_user_module,
                "get_relational_engine",
                return_value=self._engine_returning(existing_user),
            ),
            patch.object(get_default_user_module, "create_default_user", new=create_mock),
        ):
            returned = await get_default_user_module.get_default_user()

        get_base_config.cache_clear()
        assert returned is existing_user
        create_mock.assert_not_awaited()


class TestLoggingBelongsToTheServer:
    """The login warning is a server concern, not a library one (SDK-549).

    ``create_default_user`` runs under plain SDK and CLI use too, where nobody
    is going to log into this account, so it must stay silent. The API server
    reports the posture at startup instead. Superuser status is documented, not
    logged.
    """

    @pytest.mark.asyncio
    async def test_create_default_user_logs_nothing(self, created_user_password, caplog):
        import logging

        with caplog.at_level(logging.DEBUG):
            await created_user_password(None)

        assert caplog.records == [], (
            "library-level creation must not log: plain SDK scripts would see it on "
            "every fresh install"
        )

    def test_server_warns_only_when_login_is_impossible(self, monkeypatch):
        # startup_checks, not client: importing the FastAPI app here is
        # order-dependent (a sibling test shadows
        # cognee.modules.users.methods.get_authenticated_user with its module).
        from cognee.api.startup_checks import report_default_user_login_posture

        warnings = []
        monkeypatch.setattr(
            "cognee.api.startup_checks.logger",
            type("L", (), {"warning": lambda _self, msg, *a: warnings.append(msg)})(),
        )

        monkeypatch.delenv("DEFAULT_USER_PASSWORD", raising=False)
        get_base_config.cache_clear()
        report_default_user_login_posture()
        assert len(warnings) == 1
        assert "cannot be logged into" in warnings[0]
        # Privilege level is documented in the docstring and .env.template only.
        assert "superuser" not in warnings[0].lower()

        monkeypatch.setenv("DEFAULT_USER_PASSWORD", "operator-chosen-password")
        get_base_config.cache_clear()
        report_default_user_login_posture()
        assert len(warnings) == 1, "a configured password is the normal case: stay quiet"

        get_base_config.cache_clear()

    def test_superuser_status_is_documented(self):
        """Removing it from the log must not remove it from the docs."""
        from pathlib import Path

        docstring = create_default_user.__doc__ or ""
        assert "superuser" in docstring.lower()

        env_template = Path(__file__).parents[5] / ".env.template"
        default_user_block = env_template.read_text().split("-- Default user")[1][:900]
        assert "SUPERUSER" in default_user_block
