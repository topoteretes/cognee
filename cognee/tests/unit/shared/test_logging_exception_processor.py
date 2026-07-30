"""Regression tests for the exception_handler structlog processor in setup_logging().

The processor once checked `hasattr(exc_type, __name__)` — the bare module
variable instead of the string literal "__name__" — so the check compared
against "cognee.shared.logging_utils" and was always False, silently dropping
exception_type from every structured log record. These tests pin the contract.
"""

import sys

import structlog

from cognee.shared.logging_utils import setup_logging


def _get_exception_handler():
    """Return the exception_handler processor from the configured structlog chain."""
    setup_logging()
    processors = structlog.get_config()["processors"]
    return next(p for p in processors if getattr(p, "__name__", "") == "exception_handler")


class TestExceptionHandlerProcessor:
    def test_exception_type_populated_from_exc_info_tuple(self):
        original_hook = sys.excepthook
        try:
            handler = _get_exception_handler()

            try:
                raise ValueError("database connection failed")
            except ValueError:
                exc_info = sys.exc_info()

            event_dict = handler(None, "error", {"exc_info": exc_info})

            assert event_dict["exception_type"] == "ValueError"
            assert event_dict["exception_message"] == "database connection failed"
            assert event_dict["traceback"] is True
        finally:
            sys.excepthook = original_hook

    def test_exception_type_populated_from_exc_info_true(self):
        """exc_info=True must resolve the active exception via sys.exc_info()."""
        original_hook = sys.excepthook
        try:
            handler = _get_exception_handler()

            try:
                raise RuntimeError("boom")
            except RuntimeError:
                event_dict = handler(None, "error", {"exc_info": True})

            assert event_dict["exception_type"] == "RuntimeError"
            assert event_dict["exception_message"] == "boom"
        finally:
            sys.excepthook = original_hook

    def test_event_dict_without_exc_info_is_untouched(self):
        original_hook = sys.excepthook
        try:
            handler = _get_exception_handler()

            event_dict = handler(None, "info", {"event": "all good"})

            assert "exception_type" not in event_dict
            assert "exception_message" not in event_dict
            assert "traceback" not in event_dict
        finally:
            sys.excepthook = original_hook
