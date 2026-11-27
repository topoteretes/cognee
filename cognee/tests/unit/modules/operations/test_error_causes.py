from cognee.modules.operations.scrub_error import ERROR_MESSAGE_MAX_LENGTH, scrub_error_message


def test_empty_timeout_keeps_its_class():
    assert scrub_error_message(TimeoutError()) == "TimeoutError"


def test_explicit_cause_is_scrubbed():
    error = RuntimeError("Graph edge indexing error")
    error.__cause__ = ConnectionError(
        "postgresql://alice:my-password@db/test password=other-secret"
    )
    message = scrub_error_message(error)
    assert "ConnectionError" in message
    assert "my-password" not in message
    assert "other-secret" not in message


def test_cause_cycle_and_large_message_are_bounded():
    error = RuntimeError("outer")
    cause = RuntimeError("x" * 10000)
    error.__cause__ = cause
    cause.__cause__ = error
    assert len(scrub_error_message(error)) <= ERROR_MESSAGE_MAX_LENGTH


def test_implicit_context_is_not_exposed():
    error = RuntimeError("current failure")
    error.__context__ = RuntimeError("unrelated private data")
    assert scrub_error_message(error) == "current failure"
