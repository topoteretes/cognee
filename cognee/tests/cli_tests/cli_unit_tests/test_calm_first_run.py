"""Unit tests for the calm-first-run CLI toolkit: capability probe, stage
board degradation, error anatomy, preflight, diagnostics mapping, and the
dormant hint engine's suppression matrix."""

import io
import json
import logging
import sys

import pytest

from cognee.cli import diagnostics, ui
from cognee.cli.preflight import (
    PreflightError,
    needs_for_search_type,
    run_preflight,
)


# --- capability probe --------------------------------------------------------


def test_no_color_kills_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    caps = ui.detect_caps()
    assert caps.color is False


def test_ci_forces_plain_progress(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("COGNEE_PROGRESS", raising=False)
    caps = ui.detect_caps()
    assert caps.ci is True
    assert caps.progress in ("plain", "off")


def test_progress_env_off(monkeypatch):
    monkeypatch.setenv("COGNEE_PROGRESS", "off")
    assert ui.detect_caps().progress == "off"


def test_quiet_wins(monkeypatch):
    monkeypatch.setenv("COGNEE_PROGRESS", "live")
    assert ui.detect_caps(quiet=True).progress == "off"


# --- glyphs / style -----------------------------------------------------------


def test_ascii_floor():
    glyphs = ui.Glyphs(unicode_ok=False)
    for symbol in (glyphs.ok, glyphs.fail, glyphs.pending, glyphs.bullet, *glyphs.spinner):
        symbol.encode("ascii")  # must never raise


def test_style_disabled_emits_no_ansi():
    style = ui.Style(enabled=False)
    assert style.red("x") == "x"
    assert style.bold("x") == "x"


def test_truncate_never_wraps():
    long_line = "x" * 500
    assert len(ui._truncate(long_line, 80)) <= 80


def test_format_duration_shapes():
    assert ui.format_duration(0.298) == "298ms"
    assert ui.format_duration(2.13) == "2.1s"
    assert ui.format_duration(102) == "1m 42s"
    assert ui.format_duration(3700) == "1h 01m"


# --- stage board (plain mode) -------------------------------------------------


def _plain_caps():
    return ui.TermCaps(
        stdout_tty=False,
        stderr_tty=False,
        color=False,
        unicode=True,
        ci=True,
        width=80,
        progress="plain",
    )


def test_stage_board_plain_is_append_only(capsys):
    board = ui.StageBoard("Cognifying test", caps=_plain_caps(), known_stages=ui.COGNIFY_STAGES)
    board.start()
    board.stage_started("classify_documents")
    board.stage_completed("classify_documents")
    board.finish("Cognified test in 1.0s", next_command="cognee-cli search")
    err = capsys.readouterr().err
    assert "Cognifying test" in err
    assert "Classifying documents..." in err
    assert "Classified documents" in err
    assert "Cognified test in 1.0s" in err
    assert "\033[" not in err  # no ANSI in plain mode
    assert "\r" not in err  # no in-place redraws in plain mode


def test_stage_board_events_via_logging(capsys):
    caps = _plain_caps()
    with ui.pipeline_progress("T", known_stages=ui.COGNIFY_STAGES, caps=caps):
        logger = logging.getLogger("run_tasks_base")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            logger.info("Coroutine task started: `extract_chunks_from_documents`")
            logger.info("Coroutine task completed: `extract_chunks_from_documents`")
        finally:
            logger.setLevel(original_level)
    err = capsys.readouterr().err
    assert "Extracting chunks..." in err
    assert "Extracted chunks" in err


def test_unknown_task_names_are_prettified(capsys):
    caps = _plain_caps()
    with ui.pipeline_progress("T", caps=caps) as board:
        handler = ui._TaskEventHandler(board)
        record = logging.LogRecord(
            "run_tasks_base",
            logging.INFO,
            __file__,
            1,
            "Coroutine task started: `my_custom_task`",
            (),
            None,
        )
        handler.emit(record)
    err = capsys.readouterr().err
    assert "my custom task" in err.lower()


def test_hidden_tasks_stay_hidden(capsys):
    caps = _plain_caps()
    with ui.pipeline_progress("T", caps=caps) as board:
        handler = ui._TaskEventHandler(board)
        record = logging.LogRecord(
            "run_tasks_base",
            logging.INFO,
            __file__,
            1,
            "Coroutine task started: `check_permissions_on_dataset`",
            (),
            None,
        )
        handler.emit(record)
    assert "permissions" not in capsys.readouterr().err.lower()


def test_structlog_dict_records_are_parsed(capsys):
    caps = _plain_caps()
    with ui.pipeline_progress("T", known_stages=ui.COGNIFY_STAGES, caps=caps) as board:
        handler = ui._TaskEventHandler(board)
        record = logging.LogRecord(
            "run_tasks_base",
            logging.INFO,
            __file__,
            1,
            {"event": "Coroutine task started: `classify_documents`"},
            (),
            None,
        )
        handler.emit(record)
    assert "Classifying documents..." in capsys.readouterr().err


# --- error anatomy ------------------------------------------------------------


def test_error_block_renders_fix_lines(capsys):
    caps = _plain_caps()
    ui.error_block(
        "LLM_API_KEY is not set",
        why="cognee needs a key.",
        fixes=[("Fix", "export LLM_API_KEY=sk-..."), ("Check", "cognee-cli doctor")],
        footer="(stopped before any work ran)",
        caps=caps,
    )
    err = capsys.readouterr().err
    assert "LLM_API_KEY is not set" in err
    assert "export LLM_API_KEY=sk-..." in err
    assert "cognee-cli doctor" in err
    assert "(stopped before any work ran)" in err


def test_guide_block_teaches_the_rail(capsys):
    ui.guide_block(
        "Your memory is empty — nothing has been added yet.",
        ["cognee-cli add <file>", "cognee-cli cognify"],
        caps=_plain_caps(),
    )
    err = capsys.readouterr().err
    assert "memory is empty" in err
    assert "cognee-cli add" in err


# --- diagnostics mapping ------------------------------------------------------


def test_diagnostics_maps_missing_key():
    from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError

    try:
        raise LLMAPIKeyNotSetError()
    except Exception as error:
        calm = diagnostics.describe_exception(error)
    assert calm is not None
    assert "LLM_API_KEY" in calm.title
    assert any("export LLM_API_KEY" in text for _label, text in calm.fixes)


def test_diagnostics_walks_wrapped_chains():
    """Command wrappers bury the real cause mid-chain — matching must not
    only look at the deepest link."""

    class AuthenticationError(Exception):
        pass

    try:
        try:
            try:
                raise AuthenticationError("Incorrect API key provided: sk-bad")
            except Exception as inner:
                raise RuntimeError(f"Failed to add data: {inner}") from inner
        except Exception as middle:
            raise RuntimeError(str(middle)) from middle
    except Exception as outer:
        calm = diagnostics.describe_exception(outer)
    assert calm is not None
    assert "rejected your API key" in calm.title


def test_diagnostics_maps_missing_extra():
    error = ModuleNotFoundError("No module named 'anthropic'", name="anthropic")
    calm = diagnostics.describe_exception(error)
    assert calm is not None
    assert 'pip install "cognee[anthropic]"' in dict(calm.fixes).get("Fix", "")


def test_diagnostics_unknown_returns_none():
    assert diagnostics.describe_exception(ValueError("mystery")) is None


# --- preflight ----------------------------------------------------------------


def test_preflight_missing_key_fails_fast(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", str(tmp_path / "data"))
    monkeypatch.setenv("SYSTEM_ROOT_DIRECTORY", str(tmp_path / "system"))

    from cognee.infrastructure.llm import config as llm_config_module

    class FakeConfig:
        llm_provider = "openai"
        llm_model = "openai/gpt-5-mini"
        llm_api_key = None
        llm_endpoint = ""

    monkeypatch.setattr(llm_config_module, "get_llm_config", lambda: FakeConfig())
    with pytest.raises(PreflightError) as excinfo:
        run_preflight(need_llm=True, need_embeddings=False)
    assert "LLM_API_KEY is not set" in excinfo.value.calm.title


def test_preflight_placeholder_key_detected(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", str(tmp_path / "data"))
    monkeypatch.setenv("SYSTEM_ROOT_DIRECTORY", str(tmp_path / "system"))

    from cognee.infrastructure.llm import config as llm_config_module

    class FakeConfig:
        llm_provider = "openai"
        llm_model = "openai/gpt-5-mini"
        llm_api_key = "your_api_key"
        llm_endpoint = ""

    monkeypatch.setattr(llm_config_module, "get_llm_config", lambda: FakeConfig())
    with pytest.raises(PreflightError) as excinfo:
        run_preflight(need_llm=True, need_embeddings=False)
    assert "placeholder" in excinfo.value.calm.title


def test_search_type_needs():
    assert needs_for_search_type("GRAPH_COMPLETION") == (True, True)
    assert needs_for_search_type("CHUNKS") == (False, True)
    assert needs_for_search_type("CYPHER") == (False, False)


# --- hints (must ship dormant and suppressible) --------------------------------


def test_hints_suppressed_in_ci(monkeypatch, tmp_path):
    from cognee.cli import hints

    monkeypatch.setenv("COGNEE_CLI_STATE", str(tmp_path))
    monkeypatch.setenv("CI", "true")
    assert hints.emit_hint("test.hint", ["hello"]) is False


def test_hints_suppressed_by_env(monkeypatch, tmp_path):
    from cognee.cli import hints

    monkeypatch.setenv("COGNEE_CLI_STATE", str(tmp_path))
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("COGNEE_NO_HINTS", "1")
    assert hints.emit_hint("test.hint", ["hello"]) is False


def test_hint_lifetime_once(monkeypatch, tmp_path):
    from cognee.cli import hints

    monkeypatch.setenv("COGNEE_CLI_STATE", str(tmp_path))
    for var in ("CI", "COGNEE_NO_HINTS", "TERM"):
        monkeypatch.delenv(var, raising=False)

    caps = ui.TermCaps(
        stdout_tty=True,
        stderr_tty=True,
        color=False,
        unicode=True,
        ci=False,
        width=80,
        progress="live",
    )
    assert hints.emit_hint("test.once", ["hello"], caps=caps) is True
    assert hints.emit_hint("test.once", ["hello"], caps=caps) is False  # lifetime cap
    assert hints.emit_hint("test.other", ["hello"], caps=caps) is False  # daily cap


def test_record_event_counts(monkeypatch, tmp_path):
    from cognee.cli import hints

    monkeypatch.setenv("COGNEE_CLI_STATE", str(tmp_path))
    hints.record_event("cognify_success")
    state = hints.record_event("cognify_success")
    assert state["counters"]["cognify_success"] == 2
    assert "first_run_at" in state


# --- empty-state check (relational, never a graph probe) -----------------------


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)


def test_memory_state_empty(monkeypatch):
    from cognee.cli import empty_state
    from cognee.modules.data import methods as data_methods

    async def no_datasets(user_id):
        return []

    monkeypatch.setattr(data_methods, "get_datasets", no_datasets)
    state, name, count = _run_async(empty_state.check_memory_state(type("U", (), {"id": 1})()))
    assert state == "empty"


def test_memory_state_not_cognified_and_ready(monkeypatch):
    """State is answered from pipeline-run history in the relational DB —
    NOT a graph-engine probe, which under multi-tenant access control would
    look at the wrong per-dataset database and block search forever."""
    from cognee.cli import empty_state
    from cognee.modules.data import methods as data_methods
    from cognee.modules.pipelines.operations import get_pipeline_status as status_module

    dataset = type("D", (), {"id": 1, "name": "docs"})()

    async def one_dataset(user_id):
        return [dataset]

    async def three_docs(dataset_id):
        return [1, 2, 3]

    monkeypatch.setattr(data_methods, "get_datasets", one_dataset)
    monkeypatch.setattr(data_methods, "get_dataset_data", three_docs)

    runs = {}

    async def fake_status(dataset_ids, pipeline_name):
        return runs

    monkeypatch.setattr(status_module, "get_pipeline_status", fake_status)

    # no cognify run ever -> teach cognify
    state, name, count = _run_async(empty_state.check_memory_state(type("U", (), {"id": 1})()))
    assert (state, name, count) == ("not_cognified", "docs", 3)

    # errored run only -> still teach cognify
    runs["1"] = "PipelineRunStatus.DATASET_PROCESSING_ERRORED"
    state, _, _ = _run_async(empty_state.check_memory_state(type("U", (), {"id": 1})()))
    assert state == "not_cognified"

    # completed run -> ready
    runs["1"] = "PipelineRunStatus.DATASET_PROCESSING_COMPLETED"
    state, _, _ = _run_async(empty_state.check_memory_state(type("U", (), {"id": 1})()))
    assert state == "ready"
