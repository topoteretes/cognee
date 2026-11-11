"""``with_retries``: one connector request, retried by the connector's policy, with the host's
authorization checked before every attempt and every wait."""

import pytest

from cognee.tasks.ingestion.dlt_utils import with_retries

WITH_RETRIES = with_retries


class _Transient(Exception):
    pass


class _Revoked(Exception):
    pass


def _recorder(*outcomes):
    """``attempt`` answering ``outcomes`` in turn (an exception is raised), and the event log."""
    events, answers = [], list(outcomes)

    def attempt():
        events.append("attempt")
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def check_active():
        events.append("check")

    def sleep(seconds):
        events.append(f"sleep {seconds}")

    return attempt, check_active, sleep, events


def _run(attempt, check_active, sleep, *, attempts=3, delay=lambda error, number: 2**number):
    return WITH_RETRIES(
        attempt,
        retry_on=(_Transient,),
        delay=delay,
        attempts=attempts,
        check_active=check_active,
        sleep=sleep,
    )


def test_with_retries_checks_before_every_attempt_and_every_wait():
    attempt, check, sleep, events = _recorder(_Transient(), _Transient(), "ok")

    assert _run(attempt, check, sleep) == "ok"

    assert events == [
        "check",
        "attempt",
        "check",
        "sleep 1",
        "check",
        "attempt",
        "check",
        "sleep 2",
        "check",
        "attempt",
    ]


def test_with_retries_stops_when_the_check_fails_before_a_wait():
    """A revoked run does not wait out a backoff, and its error is not a provider failure."""
    attempt, _, sleep, events = _recorder(_Transient(), "ok")
    checks = iter([None, _Revoked()])

    def check_active():
        failure = next(checks)
        if failure is not None:
            raise failure

    with pytest.raises(_Revoked):
        _run(attempt, check_active, sleep)

    assert events == ["attempt"]


def test_with_retries_raises_an_error_the_connector_does_not_retry_at_once():
    attempt, check, sleep, events = _recorder(ValueError("bad request"), "ok")

    with pytest.raises(ValueError):
        _run(attempt, check, sleep)

    assert events == ["check", "attempt"]


def test_with_retries_raises_when_the_policy_declines_the_wait():
    """A provider window longer than the connector waits out is raised, not slept."""
    attempt, check, sleep, events = _recorder(_Transient(), "ok")

    with pytest.raises(_Transient):
        _run(attempt, check, sleep, delay=lambda error, number: None)

    assert events == ["check", "attempt"]


def test_with_retries_raises_the_last_attempts_error():
    attempt, check, sleep, events = _recorder(_Transient("1"), _Transient("2"))

    with pytest.raises(_Transient, match="2"):
        _run(attempt, check, sleep, attempts=2)

    assert events == ["check", "attempt", "check", "sleep 1", "check", "attempt"]


def test_with_retries_never_retries_the_checks_own_error():
    """The check sits outside the retried call, even when it raises a retried class."""
    attempt, _, sleep, events = _recorder("ok")

    def check_active():
        raise _Transient("revoked")

    with pytest.raises(_Transient, match="revoked"):
        _run(attempt, check_active, sleep)

    assert events == []


def test_with_retries_runs_without_a_check():
    attempt, _, sleep, events = _recorder(_Transient(), "ok")

    assert _run(attempt, None, sleep) == "ok"
    assert events == ["attempt", "sleep 1", "attempt"]
