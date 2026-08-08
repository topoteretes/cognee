"""Regression tests for the uncaught-exception hook installed by setup_logging().

The hook was once silently dropped during a refactor — handle_exception was
defined but never assigned to sys.excepthook, disabling structlog handling of
uncaught exceptions without any test noticing. These tests pin the contract.
"""

import sys

from cognee.shared.logging_utils import setup_logging


class TestExcepthookInstallation:
    def test_setup_logging_installs_custom_excepthook(self):
        original_hook = sys.excepthook
        try:
            setup_logging()
            assert sys.excepthook is not sys.__excepthook__, (
                "setup_logging() must install a custom sys.excepthook so uncaught "
                "exceptions are logged through structlog."
            )
            assert sys.excepthook.__qualname__.endswith("handle_exception")
        finally:
            sys.excepthook = original_hook

    def test_handler_falls_back_to_plain_traceback_when_rendering_raises(self, monkeypatch):
        """If structlog/rich rendering raises, the handler must still print a
        plain traceback instead of dying inside the hook."""
        import structlog

        original_hook = sys.excepthook
        try:
            setup_logging()
            handler = sys.excepthook

            class ExplodingLogger:
                def error(self, *args, **kwargs):
                    raise RuntimeError("renderer exploded")

            monkeypatch.setattr(structlog, "get_logger", lambda *a, **k: ExplodingLogger())

            try:
                raise ValueError("the original error")
            except ValueError:
                exc_type, exc_value, tb = sys.exc_info()

            # Capture with redirect_stdout/stderr (StringIO), NOT capsys/capfd:
            # setup_logging() enables colored output, so on Windows colorama is
            # init'd with strip=False (force_colors) and routes writes through the
            # Win32 console API — bypassing both capsys (Python streams) and capfd
            # (fds). Replacing the streams with StringIO sidesteps colorama entirely.
            import io
            from contextlib import redirect_stderr, redirect_stdout

            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                handler(exc_type, exc_value, tb)

            combined = out.getvalue() + err.getvalue()
            assert "the original error" in combined
            assert "plain traceback" in combined.lower()
        finally:
            sys.excepthook = original_hook

    def test_keyboard_interrupt_passes_through(self, monkeypatch):
        """KeyboardInterrupt must go straight to the default hook, unlogged."""
        import structlog

        original_hook = sys.excepthook
        try:
            setup_logging()
            handler = sys.excepthook

            logged = []

            class RecordingLogger:
                def error(self, *args, **kwargs):
                    logged.append((args, kwargs))

            monkeypatch.setattr(structlog, "get_logger", lambda *a, **k: RecordingLogger())

            passed_through = []
            monkeypatch.setattr(sys, "__excepthook__", lambda *args: passed_through.append(args))

            handler(KeyboardInterrupt, KeyboardInterrupt(), None)

            assert logged == []
            assert len(passed_through) == 1
        finally:
            sys.excepthook = original_hook
