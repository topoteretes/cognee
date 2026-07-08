"""Unit tests for cognee-cli doctor: check ordering, offline skip, json
schema, and exit-code behavior (0 = healthy/notes, 1 = any failure)."""

import argparse
import json

import pytest

from cognee.cli.commands.doctor_command import CheckResult, DoctorCommand, _mask_key


def _args(**overrides):
    defaults = {"verbose": False, "offline": True, "json": False}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def _fresh_configs():
    """Configs are lru_cached; monkeypatched env vars only take effect if the
    caches are cleared around each test."""

    def clear_all():
        for path, name in [
            ("cognee.infrastructure.llm.config", "get_llm_config"),
            ("cognee.base_config", "get_base_config"),
            ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
            ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
            ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
            ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ]:
            try:
                module = __import__(path, fromlist=[name])
                getattr(module, name).cache_clear()
            except Exception:
                pass

    clear_all()
    yield
    clear_all()


def test_mask_key_keeps_prefix_and_tail():
    assert _mask_key("sk-proj-abcdefghijklw9Az") == "sk-****w9Az"
    assert _mask_key("short") == "sh****"
    assert _mask_key("") == ""


def test_offline_run_orders_checks_and_skips_network(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", str(tmp_path / "data"))
    monkeypatch.setenv("SYSTEM_ROOT_DIRECTORY", str(tmp_path / "system"))
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key-abcd1234")

    command = DoctorCommand()
    command.execute(_args())
    out = capsys.readouterr().out

    python_pos = out.find("Python")
    llm_pos = out.find("LLM")
    storage_pos = out.find("Storage")
    assert -1 < python_pos < llm_pos < storage_pos
    assert "--offline" in out  # network checks visibly skipped, not silently dropped


def test_json_output_schema(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", str(tmp_path / "data"))
    monkeypatch.setenv("SYSTEM_ROOT_DIRECTORY", str(tmp_path / "system"))
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key-abcd1234")

    command = DoctorCommand()
    command.execute(_args(json=True))
    payload = json.loads(capsys.readouterr().out)

    assert {"checks", "summary"} <= set(payload)
    assert payload["summary"]["total"] == len(payload["checks"])
    for check in payload["checks"]:
        assert {"id", "status", "summary"} <= set(check)
        assert check["status"] in ("ok", "note", "fail", "skip")


def test_failure_sets_exit_code_1(monkeypatch, capsys):
    command = DoctorCommand()

    async def fake_checks(offline):
        return [
            CheckResult("python", "ok", "Python 3.12"),
            CheckResult("llm_config", "fail", "LLM_API_KEY is not set", fix="export ..."),
        ]

    monkeypatch.setattr(command, "_run_checks", fake_checks)
    with pytest.raises(SystemExit) as excinfo:
        command.execute(_args())
    assert excinfo.value.code == 1
    assert "Doctor found issues" in capsys.readouterr().out


def test_notes_do_not_fail(monkeypatch, capsys):
    command = DoctorCommand()

    async def fake_checks(offline):
        return [
            CheckResult("python", "ok", "Python 3.12"),
            CheckResult("graph", "note", "Graph is empty", fix="Run: cognee-cli add"),
        ]

    monkeypatch.setattr(command, "_run_checks", fake_checks)
    command.execute(_args())  # must not raise SystemExit
    out = capsys.readouterr().out
    assert "ready to cognify" in out
