"""Unit tests for `cognee-cli doctor`.

The command must aggregate three sections (config consistency, service
health, optional live probes) into a single pass/fail with a non-zero exit
on any failure. Provider checks and the health checker are stubbed so the
tests are hermetic — no databases or network.
"""

import argparse
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.cli.commands.doctor_command import DoctorCommand
from cognee.cli.exceptions import CliCommandException


def _healthy_response():
    from cognee.api.v1.health.health import ComponentHealth, HealthResponse, HealthStatus

    component = ComponentHealth(
        status=HealthStatus.HEALTHY, provider="sqlite", response_time_ms=1, details="ok"
    )
    return HealthResponse(
        status=HealthStatus.HEALTHY,
        timestamp="now",
        version="0.0.0",
        uptime=1,
        components={"relational_db": component},
    )


def _unhealthy_response():
    from cognee.api.v1.health.health import ComponentHealth, HealthResponse, HealthStatus

    component = ComponentHealth(
        status=HealthStatus.UNHEALTHY,
        provider="sqlite",
        response_time_ms=0,
        details="unable to open database file",
    )
    return HealthResponse(
        status=HealthStatus.UNHEALTHY,
        timestamp="now",
        version="0.0.0",
        uptime=1,
        components={"relational_db": component},
    )


@pytest.fixture
def stub_configs(monkeypatch):
    llm_config = SimpleNamespace(
        llm_provider="openai",
        llm_model="openai/gpt-5-mini",
        llm_api_key="sk-test",
        llm_azure_use_managed_identity=False,
    )
    embedding_config = SimpleNamespace(
        embedding_provider="openai",
        embedding_model="openai/text-embedding-3-large",
        embedding_api_key=None,
        embedding_endpoint=None,
        embedding_dimensions=3072,
    )
    import cognee.infrastructure.llm.config as llm_config_module
    import cognee.infrastructure.databases.vector.embeddings.config as embedding_config_module

    monkeypatch.setattr(llm_config_module, "get_llm_context_config", lambda: llm_config)
    monkeypatch.setattr(
        embedding_config_module, "get_embedding_context_config", lambda: embedding_config
    )
    return llm_config, embedding_config


def _make_args(probe=False):
    return argparse.Namespace(probe=probe)


class TestDoctorCommand:
    def test_parser_registers_probe_flag(self):
        parser = argparse.ArgumentParser()
        DoctorCommand().configure_parser(parser)
        args = parser.parse_args(["--probe"])
        assert args.probe is True
        assert parser.parse_args([]).probe is False

    def test_all_healthy_exits_cleanly(self, monkeypatch, stub_configs):
        import cognee.api.v1.health.health as health_module

        monkeypatch.setattr(
            health_module.health_checker,
            "get_health_status",
            AsyncMock(return_value=_healthy_response()),
        )
        DoctorCommand().execute(_make_args())

    def test_config_problem_fails_with_nonzero_exit(self, monkeypatch, stub_configs):
        llm_config, _ = stub_configs
        llm_config.llm_provider = "anthropic"  # trips the only-LLM-configured trap
        import cognee.api.v1.health.health as health_module

        monkeypatch.setattr(
            health_module.health_checker,
            "get_health_status",
            AsyncMock(return_value=_healthy_response()),
        )
        with pytest.raises(CliCommandException) as exc_info:
            DoctorCommand().execute(_make_args())
        assert exc_info.value.error_code == 1

    def test_unhealthy_service_fails_with_nonzero_exit(self, monkeypatch, stub_configs):
        import cognee.api.v1.health.health as health_module

        monkeypatch.setattr(
            health_module.health_checker,
            "get_health_status",
            AsyncMock(return_value=_unhealthy_response()),
        )
        with pytest.raises(CliCommandException) as exc_info:
            DoctorCommand().execute(_make_args())
        assert exc_info.value.error_code == 1

    def test_health_check_error_logs_are_suppressed_then_restored(self, monkeypatch, stub_configs):
        """The health checker logs failures at ERROR with full tracebacks; the
        doctor reports them as ✗ lines instead, so those logs must be disabled
        while the check runs — and re-enabled once doctor is done."""
        import logging

        import cognee.api.v1.health.health as health_module

        disable_level_during_check = []

        async def record_and_fail(detailed=False):
            disable_level_during_check.append(logging.root.manager.disable)
            return _unhealthy_response()

        monkeypatch.setattr(health_module.health_checker, "get_health_status", record_and_fail)
        with pytest.raises(CliCommandException):
            DoctorCommand().execute(_make_args())
        assert disable_level_during_check == [logging.ERROR]
        assert logging.root.manager.disable == logging.NOTSET
