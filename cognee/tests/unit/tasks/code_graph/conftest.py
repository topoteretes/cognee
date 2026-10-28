from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def keep_newlines_as_written(monkeypatch):
    """Windows would write ``\\n`` as ``\\r\\n``; these tests assert on the exact text."""
    write_text = Path.write_text
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, data, encoding=None, errors=None, newline="": write_text(
            self, data, encoding, errors, newline
        ),
    )
