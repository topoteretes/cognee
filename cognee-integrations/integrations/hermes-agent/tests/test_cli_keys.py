"""``hermes cognee keys`` — list and revoke the per-machine keys of plugin agents.

Hermes registers as the principal and provisions no agent of its own, so the
command manages the keys of every cognee plugin agent (Claude Code, Codex, …)
of the user: a plugin agent full of keys in use refuses a new machine a key
(cognee#6113), and this is how a slot is freed from Hermes.
"""

from __future__ import annotations

import argparse
import json

import pytest
from cognee_integration_hermes import cli
from cognee_integration_hermes.http_backend import CogneeHttpError


class StubBackend:
    """Answers scripted (method, path) pairs and records every request."""

    def __init__(self, responses):
        self.responses = responses
        self.requests = []
        self.closed = False

    def _request(self, method, path, *, timeout, **_kwargs):
        self.requests.append((method, path))
        outcome = self.responses.get((method, path), CogneeHttpError(404, "not found"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self, **_kwargs):
        self.closed = True


def _parse(argv):
    parser = argparse.ArgumentParser()
    cli.register_cli(parser.add_subparsers(dest="command").add_parser("cognee"))
    return parser.parse_args(["cognee", *argv])


def _run(monkeypatch, backend, argv):
    monkeypatch.setattr(cli, "_connect_backend", lambda: backend)
    with pytest.raises(SystemExit) as exited:
        cli.cognee_command(_parse(argv))
    return exited.value.code


STATUS = {
    "integrations": [],
    "plugins": [{"key": "claude-code"}, {"key": "codex"}, {"key": "openclaw"}],
}
CLAUDE_KEYS = {
    "keys": [
        {
            "id": "k-old",
            "label": "aaaa****",
            "createdAt": "2026-08-01T10:00:00Z",
            "lastUsedAt": "2026-09-20T09:00:00Z",
        },
        {
            "id": "k-new",
            "label": "bbbb****",
            "createdAt": "2026-09-01T10:00:00Z",
            "lastUsedAt": "2026-10-20T09:00:00Z",
        },
        {
            "id": "k-unused",
            "label": "cccc****",
            "createdAt": "2026-10-19T10:00:00Z",
            "lastUsedAt": None,
        },
    ]
}


def test_list_covers_every_plugin_agent_sorted_by_last_use(monkeypatch, capsys):
    backend = StubBackend(
        {
            ("GET", "/api/v1/integrations/status"): STATUS,
            ("GET", "/api/v1/integrations/plugins/claude-code/keys"): CLAUDE_KEYS,
            ("GET", "/api/v1/integrations/plugins/codex/keys"): {
                "keys": [
                    {
                        "id": "codex-1",
                        "createdAt": "2026-10-01T00:00:00Z",
                        "lastUsedAt": "2026-10-10T00:00:00Z",
                    }
                ]
            },
            # openclaw never provisioned: the server answers an empty listing.
            ("GET", "/api/v1/integrations/plugins/openclaw/keys"): {"keys": []},
        }
    )
    assert _run(monkeypatch, backend, ["keys", "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [(k["plugin"], k["id"]) for k in listing["keys"]] == [
        ("claude-code", "k-new"),
        ("codex", "codex-1"),
        ("claude-code", "k-old"),
        ("claude-code", "k-unused"),
    ]
    assert backend.closed

    assert _run(monkeypatch, StubBackend(backend.responses), ["keys", "list"]) == 0
    table = capsys.readouterr().out
    assert table.index("k-new") < table.index("k-old") < table.index("k-unused")
    assert "never" in table and "hermes cognee keys revoke" in table


def test_list_one_plugin_skips_the_status_lookup(monkeypatch, capsys):
    backend = StubBackend({("GET", "/api/v1/integrations/plugins/claude-code/keys"): CLAUDE_KEYS})
    assert _run(monkeypatch, backend, ["keys", "list", "--plugin", "claude-code", "--json"]) == 0
    assert ("GET", "/api/v1/integrations/status") not in backend.requests
    assert len(json.loads(capsys.readouterr().out)["keys"]) == 3


def test_a_server_without_key_routes_lists_nothing(monkeypatch, capsys):
    backend = StubBackend({})  # every route 404s: predates cognee#6113
    assert _run(monkeypatch, backend, ["keys", "list"]) == 0
    assert "No plugin agent keys" in capsys.readouterr().out


def test_revoke_one_key(monkeypatch, capsys):
    backend = StubBackend(
        {("DELETE", "/api/v1/integrations/plugins/claude-code/keys/k-old"): {"revoked": True}}
    )
    assert _run(monkeypatch, backend, ["keys", "revoke", "k-old", "--plugin", "claude-code"]) == 0
    assert backend.requests == [("DELETE", "/api/v1/integrations/plugins/claude-code/keys/k-old")]
    assert "Revoked key k-old" in capsys.readouterr().out
    assert backend.closed


def test_revoking_an_unknown_key_is_reported(monkeypatch, capsys):
    assert _run(monkeypatch, StubBackend({}), ["keys", "revoke", "nope", "--plugin", "codex"]) == 3
    assert "No key nope" in capsys.readouterr().out


def test_revoke_requires_the_plugin(capsys):
    with pytest.raises(SystemExit) as exited:
        _parse(["keys", "revoke", "k-old"])
    assert exited.value.code == 2
