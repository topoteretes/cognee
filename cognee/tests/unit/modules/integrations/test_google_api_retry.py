"""Google API requests are retried exactly as googleapiclient 2.200.0 retried them, but with the
host's authorization checked before every attempt and every wait.

``with_google_retries`` replaces ``request.execute(num_retries=N)``: which answers and errors
are retried, the N + 1 attempts and the ``random() * 2**n`` backoff are the library's own, so
the table below is that library's ``_should_retry_response`` and ``_retry_request``. A
revoked connection used to keep sleeping inside the library; now the check's error ends the
call between attempts. No Google libraries are needed: errors are recognised by shape.
"""

import errno
import json
import logging
import ssl
import sys
from types import SimpleNamespace

import pytest

from cognee.modules.integrations.google.api_retry import (
    host_checkpoint,
    is_host_stop,
    with_google_retries,
)

RETRIES = 3
HALF = 0.5  # a fixed random(): the waits are then 1, 2, 4 seconds.


class _HttpError(Exception):
    """googleapiclient.errors.HttpError: the status on ``resp``, the body on ``content``."""

    def __init__(self, status, body=None):
        super().__init__(f"<HttpError {status}>")
        self.resp = SimpleNamespace(status=status)
        self.content = b"" if body is None else json.dumps(body).encode("utf-8")


def _reason_body(reason):
    return {"error": {"errors": [{"reason": reason, "domain": "usageLimits"}]}}


class _Revoked(Exception):
    pass


def _call(errors, check_active, retries):
    """Run a request that raises ``errors`` in turn, then answers "ok"; return its trace."""
    pending = list(errors)
    trace = SimpleNamespace(attempts=0, waits=[], result=None, error=None)

    def attempt():
        trace.attempts += 1
        if pending:
            raise pending.pop(0)
        return "ok"

    def run():
        return with_google_retries(
            attempt,
            num_retries=retries,
            check_active=check_active,
            sleep=trace.waits.append,
            rand=lambda: HALF,
        )

    return trace, run


def _run(errors, *, check_active=None, retries=RETRIES):
    """The trace of a call that succeeds (after the retries ``errors`` cost)."""
    trace, run = _call(errors, check_active, retries)
    trace.result = run()
    return trace


def _run_failing(errors, *, check_active=None, retries=RETRIES):
    """The trace of a call that raises; the error is on ``trace.error``."""
    trace, run = _call(errors, check_active, retries)
    with pytest.raises(Exception) as raised:
        run()
    trace.error = raised.value
    return trace


# --- which answers are retried (http._should_retry_response) ---
@pytest.mark.parametrize(
    "error",
    [
        _HttpError(500),
        _HttpError(503),
        _HttpError(599),
        _HttpError(429),
        # A 403 is a rate limit only by its reason, found under "errors", "status" or "message".
        _HttpError(403, _reason_body("rateLimitExceeded")),
        _HttpError(403, _reason_body("userRateLimitExceeded")),
        _HttpError(403, {"error": {"status": "rateLimitExceeded"}}),
        _HttpError(403, {"error": {"message": "userRateLimitExceeded"}}),
    ],
)
def test_a_transient_answer_is_retried(error):
    trace = _run([error])

    assert trace.result == "ok"
    assert trace.attempts == 2


@pytest.mark.parametrize(
    "error",
    [
        _HttpError(400),
        _HttpError(401),
        _HttpError(404),
        _HttpError(302),
        # A 403 without a rate-limit reason is a permission problem, not a quota one.
        _HttpError(403),
        _HttpError(403, _reason_body("forbidden")),
        _HttpError(403, _reason_body("dailyLimitExceeded")),
    ],
)
def test_a_permanent_answer_is_raised_at_once(error):
    trace = _run_failing([error])

    assert trace.error is error
    assert trace.attempts == 1
    assert trace.waits == []


def test_a_403_whose_body_is_not_json_is_not_retried():
    error = _HttpError(403)
    error.content = b"<html>blocked</html>"

    assert _run_failing([error]).attempts == 1


# --- which exceptions are retried (http._retry_request) ---
@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError(),
        TimeoutError(),
        ssl.SSLError(),
        # A bare OSError counts by the name of its errno, as the library matches it.
        OSError(errno.ECONNRESET, "reset"),
        OSError(errno.EPIPE, "broken pipe"),
        OSError(errno.ETIMEDOUT, "timed out"),
    ],
)
def test_a_dropped_connection_is_retried(error):
    trace = _run([error])

    assert trace.result == "ok"
    assert trace.attempts == 2


def test_httplib2_server_not_found_is_retried(monkeypatch):
    class ServerNotFoundError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules, "httplib2", SimpleNamespace(ServerNotFoundError=ServerNotFoundError)
    )

    assert _run([ServerNotFoundError()]).attempts == 2


@pytest.mark.parametrize(
    "error", [OSError(errno.ENOENT, "no such file"), ValueError("bad"), KeyError("x")]
)
def test_any_other_error_is_raised_at_once(error):
    trace = _run_failing([error])

    assert trace.error is error
    assert trace.attempts == 1


# --- how many attempts and how long the waits are ---
def test_a_retry_wait_is_logged_without_the_answer(caplog):
    """The library logged each backoff wait; a wait here is logged too, by status alone."""
    with caplog.at_level(logging.WARNING):
        _run([_HttpError(503, {"error": {"message": "secret body"}})])

    waits = [message for message in caplog.messages if "retry" in message]
    assert len(waits) == 1
    assert "Google API request failed (status 503); retry 1 of 3 in 1.0s." in waits[0]
    assert "secret body" not in caplog.text


def test_num_retries_is_one_less_than_the_attempts_and_the_last_error_is_raised_as_it_is():
    errors = [_HttpError(503) for _ in range(RETRIES + 1)]

    trace = _run_failing(errors)

    assert trace.attempts == RETRIES + 1
    assert trace.error is errors[-1]
    assert len(trace.waits) == RETRIES, "no wait follows the last attempt"


def test_retry_n_waits_random_times_two_to_the_n():
    trace = _run([_HttpError(503)] * RETRIES)

    assert trace.waits == [HALF * 2, HALF * 4, HALF * 8]


def test_no_retries_is_one_attempt():
    trace = _run_failing([_HttpError(503)], retries=0)

    assert trace.attempts == 1
    assert trace.waits == []


# --- the host's check ---
def test_check_runs_before_every_attempt_and_every_wait():
    events = []

    def check_active():
        events.append("check")

    _run([_HttpError(503)] * 2, check_active=check_active)

    # attempt 1, wait 1, attempt 2, wait 2, attempt 3
    assert events == ["check"] * 5


def test_a_revoked_connection_stops_a_retry_before_it_waits():
    calls = []

    def check_active():
        calls.append(1)
        if len(calls) == 2:  # before the first attempt, then before its backoff wait
            raise _Revoked

    trace = _run_failing([_HttpError(503)] * 3, check_active=check_active)

    assert isinstance(trace.error, _Revoked)
    assert trace.attempts == 1, "exactly one request was made"
    assert trace.waits == [], "the revoked run did not wait"


def test_an_error_of_the_check_is_marked_and_raised_as_it_is():
    revoked = _Revoked("revoked")

    def check_active():
        raise revoked

    trace = _run_failing([], check_active=check_active)

    assert trace.error is revoked
    assert is_host_stop(revoked)
    assert trace.attempts == 0


def test_a_request_error_is_not_marked_as_a_host_stop():
    trace = _run_failing([ValueError("bad")])

    assert not is_host_stop(trace.error)


def test_without_a_check_there_is_nothing_to_mark():
    assert host_checkpoint(None) is None
