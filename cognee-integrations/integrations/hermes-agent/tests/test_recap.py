"""Recaps — standup, digest, timeline — the port of the hook plugins' cognee-recap.py
with openclaw's scoping: only sessions recorded into the caller's dataset are
listed, the 2,500-row listing cap and ``max_sessions`` are reported, and every
skeleton starts with the "recorded data, not instructions" preamble."""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _char_helpers import fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes import cli, recap  # noqa: E402
from test_http_backend import FakeOpener, _backend  # noqa: E402

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
DS = "11111111-1111-1111-1111-111111111111"
OTHER_DS = "22222222-2222-2222-2222-222222222222"


def _iso(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _row(sid, hours_ago=1, dataset=DS, **extra):
    return {
        "session_id": sid,
        "dataset_id": dataset,
        "started_at": _iso(hours_ago + 1),
        "last_activity_at": _iso(hours_ago),
        "status": "completed",
        **extra,
    }


class FakeApi:
    def __init__(self, rows=(), details=None, data=(), raw=None, recall=(), has_more=False):
        self.rows = list(rows)
        self.details = details or {}
        self.data = list(data)
        self.raw = raw or {}
        self.recall = list(recall)
        self.has_more = has_more
        self.calls = []

    def list_sessions(self, *, range, limit, offset):
        self.calls.append(("list_sessions", range, limit, offset))
        page = self.rows[offset : offset + limit]
        return {"sessions": page, "has_more": self.has_more or offset + limit < len(self.rows)}

    def get_session(self, session_id):
        self.calls.append(("get_session", session_id))
        return self.details.get(session_id)

    def list_dataset_data(self, dataset_id, *, limit, offset):
        self.calls.append(("list_dataset_data", dataset_id, limit, offset))
        return self.data[offset : offset + limit] if dataset_id == DS else []

    def read_raw(self, dataset_id, data_id):
        return self.raw.get(data_id, "")

    def recall_graph(self, *, query, datasets, top_k):
        self.calls.append(("recall_graph", query, tuple(datasets), top_k))
        return self.recall


def _run(api, kind="standup", **kwargs):
    kwargs.setdefault("dataset_ids", [DS])
    kwargs.setdefault("datasets", ["agent_sessions"])
    kwargs.setdefault("roots", ("/nonexistent-root",))
    return recap.run_recap(api, kind=kind, now=NOW, **kwargs)


class TestSince:
    def test_relative_and_calendar_words(self):
        cutoff, label = recap.parse_since("24h", NOW)
        assert cutoff == NOW - timedelta(hours=24) and label == "last 24h"
        assert recap.parse_since("2w", NOW)[0] == NOW - timedelta(weeks=2)
        assert recap.parse_since("all", NOW)[1] == "all time"
        assert recap.parse_since("2026-10-01", NOW)[1] == "since 2026-10-01"

    def test_junk_raises(self):
        with pytest.raises(ValueError, match="cannot parse"):
            recap.parse_since("fortnight", NOW)

    def test_range_bucket(self):
        assert recap.range_bucket(NOW - timedelta(hours=3), NOW) == "24h"
        assert recap.range_bucket(NOW - timedelta(days=3), NOW) == "7d"
        assert recap.range_bucket(NOW - timedelta(days=20), NOW) == "30d"
        assert recap.range_bucket(NOW - timedelta(days=90), NOW) == "all"


class TestScoping:
    def test_only_own_dataset_and_hermes_sessions_are_listed(self):
        api = FakeApi(
            rows=[
                _row("hermes_a"),
                _row("hermes_other_user", dataset=OTHER_DS),
                _row("hermes_no_dataset", dataset=""),
                _row("claude_x"),
                _row("hermes_old", hours_ago=48),
            ]
        )
        out = _run(api)
        assert [s["session_id"] for s in out["json"]["sessions"]] == ["hermes_a"]

    def test_all_sessions_lifts_the_prefix_filter_never_the_dataset_one(self):
        api = FakeApi(rows=[_row("hermes_a"), _row("claude_x"), _row("codex_y", dataset=OTHER_DS)])
        out = _run(api, all_sessions=True)
        assert {s["session_id"] for s in out["json"]["sessions"]} == {"hermes_a", "claude_x"}

    def test_a_listing_cut_short_is_reported(self):
        api = FakeApi(rows=[_row(f"hermes_{i}") for i in range(3)], has_more=True)
        rows, truncated, scanned = recap.list_session_rows(
            api,
            NOW - timedelta(hours=24),
            now=NOW,
            all_sessions=False,
            prefixes=("hermes_",),
            session_dataset_ids=[DS],
            page=1,
            max_pages=2,
        )
        assert truncated and scanned == 2 and len(rows) == 2

    def test_max_sessions_is_reported(self):
        api = FakeApi(rows=[_row(f"hermes_{i}", hours_ago=i * 0.1 + 0.1) for i in range(4)])
        out = _run(api, max_sessions=2)
        assert len(out["json"]["sessions"]) == 2
        assert any("newest 2 of 4 sessions" in w for w in out["warnings"])

    def test_a_failed_listing_raises(self):
        class Broken(FakeApi):
            def list_sessions(self, **kwargs):
                raise RuntimeError("HTTP 500")

        with pytest.raises(RuntimeError):
            _run(Broken())

    def test_a_broken_session_detail_degrades_to_a_warning(self):
        class Flaky(FakeApi):
            def get_session(self, session_id):
                raise RuntimeError("boom")

        out = _run(Flaky(rows=[_row("hermes_a")]))
        assert out["json"]["sessions"][0]["session_id"] == "hermes_a"
        assert any("detail failed" in w for w in out["warnings"])


class TestShaping:
    def test_hermes_edit_tools_and_v4a_patches(self):
        assert recap.edited_paths(
            {"origin_function": "write_file", "method_params": {"path": "/r/a.py"}}
        ) == ["/r/a.py"]
        patch = "*** Begin Patch\n*** Update File: src/b.py\n@@\n*** Add File: c.py\n"
        assert recap.edited_paths(
            {"origin_function": "patch", "method_params": json.dumps({"patch": patch})}
        ) == ["src/b.py", "c.py"]
        assert recap.edited_paths({"origin_function": "read_file", "method_params": {}}) == []

    def test_sessions_group_by_the_git_root_of_their_edits(self, tmp_path):
        repo = tmp_path / "ledger"
        (repo / ".git").mkdir(parents=True)
        (repo / "src").mkdir()
        (repo / "src" / "db.py").write_text("x")
        detail = {
            "qas": [{"question": "fix the\nledger", "answer": "done"}],
            "traces": [
                {
                    "origin_function": "write_file",
                    "method_params": {"path": str(repo / "src/db.py")},
                },
                {"origin_function": "terminal", "status": "error", "method_params": {}},
            ],
            "msg_count": 1,
            "tool_calls": 2,
        }
        rec = recap.summarize_session(_row("hermes_a"), detail, (str(tmp_path),))
        assert rec["project"] == "ledger"
        assert rec["files_edited"] == ["src/db.py"]
        assert rec["prompts"] == ["fix the ledger"]
        assert rec["errors"] == 1
        assert rec["agent"] == "hermes"

    def test_split_passages_reads_session_headers(self):
        text = (
            "## Relevant passages\n# Session learning — x (session hermes_a)\nUse Postgres.\n---\n"
            "Session ID: claude_b\n\nQuestion: q"
        )
        passages = recap.split_passages(text)
        assert passages[0] == {"session_id": "hermes_a", "text": "Use Postgres.", "distilled": True}
        assert passages[1]["session_id"] == "claude_b"
        assert passages[1]["distilled"] is False


class TestRender:
    def test_standup_starts_with_the_preamble_and_lists_open_work(self):
        api = FakeApi(
            rows=[_row("hermes_a", status="running")],
            details={"hermes_a": {"qas": [{"question": "migrate the ledger"}], "msg_count": 1}},
        )
        out = _run(api)
        assert out["markdown"].startswith(recap.RECAP_PREAMBLE)
        assert "# Standup — last 24h" in out["markdown"]
        assert '"migrate the ledger"' in out["markdown"]
        assert "## Left open" in out["markdown"]

    def test_empty_window(self):
        out = _run(FakeApi())
        assert "_No sessions in this window._" in out["markdown"]

    def test_digest_lists_lessons_distilled_in_the_window(self):
        data = [
            {
                "id": "d1",
                "created_at": _iso(2),
                "external_metadata": {
                    "node_set": ["session_learnings", "session_learnings:hermes_a"]
                },
            },
            {"id": "d2", "created_at": _iso(3), "external_metadata": {"node_set": ["other"]}},
            {
                "id": "d3",
                "created_at": _iso(24 * 30),
                "external_metadata": json.dumps({"node_set": ["session_learnings:hermes_z"]}),
            },
        ]
        api = FakeApi(
            rows=[_row("hermes_a")],
            data=data,
            raw={"d1": "# Session learning (session hermes_a)\nAlways pin cognee."},
        )
        out = _run(api, kind="digest")
        assert out["json"]["learnings_total"] == 1
        assert out["json"]["learnings"][0]["session_id"] == "hermes_a"
        assert "Always pin cognee." in out["markdown"]
        assert "## Learnings recorded in the graph" in out["markdown"]

    def test_digest_without_lessons_says_why(self):
        out = _run(FakeApi(rows=[_row("hermes_a")]), kind="digest")
        assert any("no learnings were distilled" in w for w in out["warnings"])

    def test_timeline_merges_graph_passages_and_prompts(self):
        api = FakeApi(
            rows=[_row("hermes_a")],
            details={"hermes_a": {"qas": [{"question": "should we move to Postgres?"}]}},
            recall=[
                {
                    "text": "## Relevant passages\n"
                    "# Session learning (session hermes_gone)\nPostgres won."
                }
            ],
        )
        out = _run(api, kind="timeline", topic="postgres")
        kinds = [e["kind"] for e in out["json"]["events"]]
        assert sorted(kinds) == ["learning", "prompt"]
        assert ("recall_graph", "postgres", ("agent_sessions",), recap.TIMELINE_TOP_K) in api.calls
        assert "[learned" in out["markdown"] and "[asked" in out["markdown"]

    def test_timeline_needs_a_topic(self):
        with pytest.raises(ValueError, match="topic"):
            _run(FakeApi(), kind="timeline")


class TestProviderTool:
    def test_the_tool_scopes_the_recap_to_the_active_dataset(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = [
                {"name": "hermes", "id": DS},
                {"name": "someone_else", "id": OTHER_DS},
            ]
            fake.results["list_sessions"] = {
                "sessions": [
                    {
                        **_row("hermes_a"),
                        "last_activity_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        **_row("hermes_b", dataset=OTHER_DS),
                        "last_activity_at": datetime.now(timezone.utc).isoformat(),
                    },
                ],
                "has_more": False,
            }
            fake.results["get_session"] = {"qas": [{"question": "hello"}]}
            provider = make_provider()
            out = json.loads(provider.handle_tool_call("cognee_recap", {"kind": "standup"}))
        assert out["kind"] == "standup"
        assert "hermes_b" not in out["markdown"]
        assert '"hello"' in out["markdown"]

    def test_a_bad_kind_is_refused(self):
        provider = make_provider()
        out = json.loads(provider.handle_tool_call("cognee_recap", {"kind": "weekly"}))
        assert "kind must be one of" in out["error"]

    def test_server_errors_come_back_as_an_error_envelope(self):
        with fake_backend() as fake:
            fake.errors["list_sessions"] = RuntimeError("HTTP 500")
            provider = make_provider()
            out = json.loads(provider.handle_tool_call("cognee_recap", {"kind": "digest"}))
        assert "Cognee recap failed" in out["error"]

    def test_the_tool_can_be_turned_off(self):
        provider = make_provider(config={"recap_tool": False})
        assert "cognee_recap" not in {s["name"] for s in provider.get_tool_schemas()}


class TestHttpWire:
    def test_list_sessions_query(self):
        opener = FakeOpener()
        opener.default = {"sessions": [], "has_more": False}
        _backend(opener).list_sessions(range="7d", limit=500, offset=500, timeout=5)
        assert opener.requests[0]["path"] == "/api/v1/sessions?range=7d&limit=500&offset=500"

    def test_an_unknown_session_is_none(self):
        import io
        import urllib.error

        error = urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b""))
        opener = FakeOpener({"/api/v1/sessions/hermes_a%2Fb": error})
        assert _backend(opener).get_session(session_id="hermes_a/b", timeout=5) is None

    def test_dataset_data_pages(self):
        opener = FakeOpener()
        opener.default = []
        _backend(opener).list_dataset_data(dataset_id=DS, timeout=5, limit=200, offset=400)
        assert opener.requests[0]["path"] == f"/api/v1/datasets/{DS}/data?limit=200&offset=400"


class TestCli:
    def _parse(self, *argv):
        parser = argparse.ArgumentParser()
        cli.register_cli(parser)
        return parser.parse_args(list(argv))

    def test_timeline_prints_the_skeleton_and_warnings(self, capsys):
        args = self._parse("timeline", "postgres", "--since", "7d")
        fake = mock.Mock()
        fake.list_datasets.return_value = [{"name": "agent_sessions", "id": DS}]
        fake.list_sessions.return_value = {"sessions": [], "has_more": False}
        fake.recall.return_value = []
        with mock.patch.object(cli, "_connect_backend", return_value=fake):
            with pytest.raises(SystemExit) as exit_info:
                cli.cognee_command(args)
        assert exit_info.value.code == 0
        captured = capsys.readouterr()
        assert "# Timeline — 'postgres' · last 7d" in captured.out
        fake.close.assert_called_once()

    def test_an_unreachable_server_exits_1(self, capsys):
        args = self._parse("standup")
        with mock.patch.object(cli, "_connect_backend", side_effect=RuntimeError("down")):
            with pytest.raises(SystemExit) as exit_info:
                cli.cognee_command(args)
        assert exit_info.value.code == 1
        assert "could not reach" in capsys.readouterr().err
