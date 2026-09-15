import pytest

from cognee.modules.users.authentication.get_auth_secret import (
    AUTH_SECRETS,
    _generated_secret,
    get_auth_secret,
    resolve_auth_secrets,
)

JWT = "FASTAPI_USERS_JWT_SECRET"
RESET = "FASTAPI_USERS_RESET_PASSWORD_TOKEN_SECRET"
VERIFY = "FASTAPI_USERS_VERIFICATION_TOKEN_SECRET"


@pytest.fixture(autouse=True)
def _fresh_generated_secrets(monkeypatch):
    for env_var in AUTH_SECRETS:
        monkeypatch.delenv(env_var, raising=False)
    _generated_secret.cache_clear()
    yield
    _generated_secret.cache_clear()


def _warnings_for(caplog, env_var):
    return [r for r in caplog.records if r.levelname == "WARNING" and env_var in r.getMessage()]


def test_configured_secret_is_returned_verbatim(monkeypatch):
    monkeypatch.setenv(JWT, "configured-secret")

    assert get_auth_secret(JWT) == "configured-secret"


def test_configured_secret_logs_nothing(monkeypatch, caplog):
    monkeypatch.setenv(JWT, "configured-secret")

    with caplog.at_level("WARNING"):
        get_auth_secret(JWT)

    assert _warnings_for(caplog, JWT) == []


def test_unset_secret_is_random_and_stable_within_the_process():
    first = get_auth_secret(JWT)
    second = get_auth_secret(JWT)

    assert first == second
    assert first != "super_secret"
    assert len(first) >= 64


def test_blank_secret_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(JWT, "   ")

    assert get_auth_secret(JWT) != "   "
    assert len(get_auth_secret(JWT)) >= 64


def test_each_variable_gets_its_own_generated_secret():
    assert get_auth_secret(JWT) != get_auth_secret(RESET)


def test_setting_the_variable_later_wins_over_the_generated_secret(monkeypatch):
    generated = get_auth_secret(JWT)

    monkeypatch.setenv(JWT, "configured-secret")

    assert get_auth_secret(JWT) == "configured-secret"
    assert generated != "configured-secret"


def test_generated_secret_warns_once_and_names_the_variable(caplog):
    with caplog.at_level("WARNING"):
        get_auth_secret(JWT)
        get_auth_secret(JWT)

    warnings = _warnings_for(caplog, JWT)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "random secret was generated" in message
    assert "more than one process" in message
    assert ".env" in message


def test_startup_resolution_warns_only_for_generated_secrets(monkeypatch, caplog):
    monkeypatch.setenv(JWT, "configured-secret")

    with caplog.at_level("WARNING"):
        resolve_auth_secrets()

    assert _warnings_for(caplog, JWT) == []
    assert len(_warnings_for(caplog, RESET)) == 1
    assert len(_warnings_for(caplog, VERIFY)) == 1


def test_startup_resolution_is_silent_when_everything_is_configured(monkeypatch, caplog):
    for env_var in AUTH_SECRETS:
        monkeypatch.setenv(env_var, f"{env_var}-value")

    with caplog.at_level("WARNING"):
        resolve_auth_secrets()

    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
