"""The WebSocket ?token= fallback must not leak into uvicorn's own logs.

Uvicorn logs the full handshake path (query string included) on every
WebSocket accept/close via the uvicorn.error logger, independent of any
reverse proxy. This pins the filter that redacts it there.
"""

import logging

import pytest

from cognee.modules.users.authentication.redact_websocket_query_secrets import (
    _RedactWebsocketTokenFilter,
)


@pytest.fixture
def redaction_filter():
    return _RedactWebsocketTokenFilter()


def _make_record(msg, args=()):
    return logging.LogRecord(
        name="uvicorn.error",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_token_in_a_percent_style_arg_is_redacted(redaction_filter):
    record = _make_record(
        '%s - "WebSocket %s" [accepted]',
        ("127.0.0.1:54321", "/api/v1/visualize/subscribe/abc?token=eyJhbGciOiJIUzI1NiJ9.secret"),
    )

    redaction_filter.filter(record)

    assert "eyJhbGciOiJIUzI1NiJ9.secret" not in record.args[1]
    assert "token=***REDACTED***" in record.args[1]
    assert record.args[1].startswith("/api/v1/visualize/subscribe/abc?")


def test_token_as_a_non_final_query_param_is_still_redacted(redaction_filter):
    record = _make_record(
        '%s - "WebSocket %s" [accepted]',
        ("127.0.0.1:54321", "/subscribe/abc?since=2026-01-01&token=secret123&other=1"),
    )

    redaction_filter.filter(record)

    assert "secret123" not in record.args[1]
    assert "since=2026-01-01" in record.args[1]
    assert "other=1" in record.args[1]


def test_a_message_with_no_token_is_left_untouched(redaction_filter):
    record = _make_record(
        '%s - "WebSocket %s" [accepted]',
        ("127.0.0.1:54321", "/subscribe/abc?since=2026-01-01"),
    )

    redaction_filter.filter(record)

    assert record.args[1] == "/subscribe/abc?since=2026-01-01"


def test_filter_always_returns_true_so_the_record_is_still_emitted(redaction_filter):
    record = _make_record("plain message with no args")

    assert redaction_filter.filter(record) is True
