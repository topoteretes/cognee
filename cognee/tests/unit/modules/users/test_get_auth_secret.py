import pytest

from cognee.modules.users.authentication.get_auth_secret import (
    _generated_secret,
    get_auth_secret,
)

ENV_VAR = "FASTAPI_USERS_JWT_SECRET"


@pytest.fixture(autouse=True)
def _fresh_generated_secrets():
    _generated_secret.cache_clear()
    yield
    _generated_secret.cache_clear()


def test_configured_secret_is_returned_verbatim(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "configured-secret")

    assert get_auth_secret(ENV_VAR) == "configured-secret"


def test_unset_secret_is_random_and_stable_within_the_process(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)

    first = get_auth_secret(ENV_VAR)
    second = get_auth_secret(ENV_VAR)

    assert first == second
    assert first != "super_secret"
    assert len(first) >= 64


def test_blank_secret_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "   ")

    assert get_auth_secret(ENV_VAR) != "   "
    assert len(get_auth_secret(ENV_VAR)) >= 64


def test_each_variable_gets_its_own_generated_secret(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("FASTAPI_USERS_RESET_PASSWORD_TOKEN_SECRET", raising=False)

    assert get_auth_secret(ENV_VAR) != get_auth_secret("FASTAPI_USERS_RESET_PASSWORD_TOKEN_SECRET")


def test_setting_the_variable_later_wins_over_the_generated_secret(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    generated = get_auth_secret(ENV_VAR)

    monkeypatch.setenv(ENV_VAR, "configured-secret")

    assert get_auth_secret(ENV_VAR) == "configured-secret"
    assert generated != "configured-secret"


def test_generated_secret_logs_a_warning_once(monkeypatch, caplog):
    monkeypatch.delenv(ENV_VAR, raising=False)

    with caplog.at_level("WARNING"):
        get_auth_secret(ENV_VAR)
        get_auth_secret(ENV_VAR)

    warnings = [r for r in caplog.records if ENV_VAR in r.getMessage()]
    assert len(warnings) == 1
