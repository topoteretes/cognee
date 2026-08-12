"""Tests for pipeline run ownership.

Ownership exists to replace a guess with a fact, so the important property is
that it only ever claims certainty when it genuinely has it. Every case where
the answer cannot be known from here must return None and hand the decision
back to the heartbeat.
"""

import os

from cognee.modules.pipelines.utils.run_ownership import (
    get_node_id,
    get_owner_pid,
    is_owner_process_alive,
)


def test_node_id_prefers_explicit_configuration(monkeypatch):
    monkeypatch.setenv("COGNEE_NODE_ID", "worker-7")
    assert get_node_id() == "worker-7"


def test_node_id_falls_back_to_hostname(monkeypatch):
    monkeypatch.delenv("COGNEE_NODE_ID", raising=False)
    assert get_node_id()


def test_this_process_is_reported_alive(monkeypatch):
    monkeypatch.setenv("COGNEE_NODE_ID", "here")
    assert is_owner_process_alive("here", get_owner_pid()) is True


def test_dead_pid_on_this_node_is_definitively_gone(monkeypatch):
    monkeypatch.setenv("COGNEE_NODE_ID", "here")

    # Find a pid that does not exist. Fork a child, reap it, reuse its pid:
    # simplest portable approach is to probe upward from an implausible value.
    dead_pid = 999_999
    while dead_pid > 0:
        try:
            os.kill(dead_pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            pass
        dead_pid -= 1

    assert is_owner_process_alive("here", dead_pid) is False


def test_another_node_is_never_judged(monkeypatch):
    """A pid from another machine's process table says nothing about this one,
    so the answer must be 'cannot tell' rather than a coin flip."""
    monkeypatch.setenv("COGNEE_NODE_ID", "here")
    assert is_owner_process_alive("elsewhere", get_owner_pid()) is None


def test_missing_ownership_is_never_judged(monkeypatch):
    monkeypatch.setenv("COGNEE_NODE_ID", "here")
    assert is_owner_process_alive(None, None) is None
    assert is_owner_process_alive("here", None) is None
    assert is_owner_process_alive(None, 1234) is None


def test_unparseable_pid_is_never_judged(monkeypatch):
    monkeypatch.setenv("COGNEE_NODE_ID", "here")
    assert is_owner_process_alive("here", "not-a-pid") is None
