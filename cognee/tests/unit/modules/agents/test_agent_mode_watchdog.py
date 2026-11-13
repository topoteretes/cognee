import signal
from unittest.mock import MagicMock, patch

import pytest

from cognee.modules.agents import agent_mode


@pytest.fixture(autouse=True)
def _reset_state():
    agent_mode._active_count = 0
    agent_mode._active_connection_ids.clear()
    agent_mode._watchdog_started = False
    if hasattr(agent_mode, "_idle_ticks"):
        agent_mode._idle_ticks = 0
    yield
    agent_mode._active_count = 0
    if hasattr(agent_mode, "_idle_ticks"):
        agent_mode._idle_ticks = 0


def _tick():
    with patch.object(agent_mode.threading, "Timer", return_value=MagicMock()):
        agent_mode._watchdog()


def test_watchdog_sends_sigterm_on_first_idle_tick():
    with patch.object(agent_mode.os, "kill") as mock_kill:
        _tick()
    mock_kill.assert_called_once_with(agent_mode.os.getpid(), signal.SIGTERM)


def test_watchdog_escalates_when_sigterm_does_not_stop_the_process():
    kill_signal = getattr(signal, "SIGKILL", None)
    with (
        patch.object(agent_mode.os, "kill") as mock_kill,
        patch.object(agent_mode.os, "_exit") as mock_exit,
    ):
        for _ in range(10):
            _tick()
    killed = [c.args[1] for c in mock_kill.call_args_list]
    escalated = (kill_signal is not None and kill_signal in killed) or mock_exit.called
    assert escalated, "watchdog kept sending SIGTERM and never escalated"


def test_watchdog_does_not_escalate_when_an_agent_registers_again():
    agent_mode._active_count = 0
    with (
        patch.object(agent_mode.os, "kill") as mock_kill,
        patch.object(agent_mode.os, "_exit") as mock_exit,
    ):
        _tick()
        agent_mode._active_count = 1
        for _ in range(10):
            _tick()
    assert [c.args[1] for c in mock_kill.call_args_list] == [signal.SIGTERM]
    mock_exit.assert_not_called()
