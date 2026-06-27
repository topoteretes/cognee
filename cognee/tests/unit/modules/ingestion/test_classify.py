"""Regression tests for ``classify`` filename derivation.

``classify`` derived a binary stream's name with ``str(data.name).split("/")[-1]``.
On Windows, ``data.name`` is a backslash path (e.g. ``C:\\dir\\file.pdf``) with no
forward slashes, so the split returned the *entire path* as the file's name --
which then propagated into the stored document name / metadata.

The fix normalizes both ``\\`` and ``/`` before taking the basename. These tests
are cross-platform strict: a revert to splitting on ``/`` only fails on any OS,
because the Windows-style input below contains backslashes regardless of host.
"""

import io
from unittest.mock import MagicMock

from cognee.modules.ingestion.classify import classify
from cognee.modules.ingestion.data_types import BinaryData


def _fake_binary_stream(name: str) -> MagicMock:
    # spec=io.BufferedReader makes isinstance(stream, BufferedReader) pass in classify.
    stream = MagicMock(spec=io.BufferedReader)
    stream.name = name
    return stream


def test_windows_path_name_is_reduced_to_basename():
    result = classify(_fake_binary_stream(r"C:\Users\foo\report.pdf"))
    assert isinstance(result, BinaryData)
    assert result.name == "report.pdf"


def test_posix_path_name_is_reduced_to_basename():
    result = classify(_fake_binary_stream("/home/user/report.pdf"))
    assert result.name == "report.pdf"


def test_explicit_filename_takes_precedence_over_stream_name():
    result = classify(_fake_binary_stream(r"C:\Users\foo\ignored.pdf"), filename="explicit.pdf")
    assert result.name == "explicit.pdf"
