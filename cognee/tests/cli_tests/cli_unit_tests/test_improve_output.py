"""``cognee-cli improve`` renders the SDK-593 status answers, not a raw dict dump."""

from types import SimpleNamespace
from unittest.mock import patch

from cognee.cli import improve_output


def _capture():
    lines: list[tuple[str, str]] = []
    patches = [
        patch.object(improve_output.fmt, name, lambda msg, _n=name: lines.append((_n, msg)))
        for name in ("success", "warning", "echo")
    ]
    return lines, patches


def test_pipeline_run_mapping_prints_per_dataset_status():
    lines, patches = _capture()
    with patches[0], patches[1], patches[2]:
        improve_output.echo_improve_result(
            {"ds-1": SimpleNamespace(status="DATASET_PROCESSING_COMPLETED")}, background=False
        )

    assert lines[0] == ("success", "Knowledge graph improved successfully!")
    assert lines[1] == ("echo", "  Dataset ds-1: DATASET_PROCESSING_COMPLETED")


def test_no_op_is_a_success_with_the_reason():
    lines, patches = _capture()
    with patches[0], patches[1], patches[2]:
        improve_output.echo_improve_result(
            {"status": "no_op", "session_ids": ["s1"], "reason": "nothing_pending"},
            background=True,
        )

    assert lines == [("success", "Nothing to improve for session(s) s1 (nothing_pending).")]


def test_busy_is_a_warning_that_says_not_to_retry():
    lines, patches = _capture()
    with patches[0], patches[1], patches[2]:
        improve_output.echo_improve_result(
            {
                "status": "busy",
                "session_ids": ["s1"],
                "session_id": "s1",
                "holder_age_seconds": 42.7,
                "rerun_requested": True,
            },
            background=False,
        )

    assert len(lines) == 1 and lines[0][0] == "warning"
    assert "session s1" in lines[0][1]
    assert "42 s" in lines[0][1]
    assert "No retry needed" in lines[0][1]


def test_accepted_lists_the_pending_stages():
    lines, patches = _capture()
    with patches[0], patches[1], patches[2]:
        improve_output.echo_improve_result(
            {
                "status": "accepted",
                "session_ids": ["s1"],
                "background": True,
                "pending_stages": ["persist_sessions", "distill_sessions"],
            },
            background=True,
        )

    expected = (
        "Improvement accepted and running in background "
        "(stages: persist_sessions, distill_sessions)."
    )
    assert lines == [("success", expected)]


def test_improve_status_only_recognises_status_shaped_dicts():
    assert improve_output.improve_status({"status": "busy"}) == "busy"
    assert improve_output.improve_status({"ds": object()}) is None
    assert improve_output.improve_status({}) is None
    assert improve_output.improve_status(None) is None
