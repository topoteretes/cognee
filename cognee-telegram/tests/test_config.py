"""Settings parsing — onboarding-critical, so the required-token error is tested."""

import pytest

from cognee_telegram.config import Settings


def test_from_env_reads_and_strips_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "  123456:ABC-DEF  ")
    settings = Settings.from_env()
    assert settings.bot_token == "123456:ABC-DEF"


def test_from_env_missing_token_raises(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        Settings.from_env()


def test_from_env_blank_token_raises(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "   ")
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        Settings.from_env()
