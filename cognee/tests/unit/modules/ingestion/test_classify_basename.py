"""Regression test for filename derivation in ``classify``.

``classify`` derived the ``BinaryData`` name with ``str(data.name).split("/")[-1]``.
On Windows, an opened file's ``.name`` is a backslash path (e.g.
``C:\\audio\\clip.mp3``) with no forward slashes, so the split kept the *entire
path* as the name instead of the basename.

The fix normalizes both ``\\`` and ``/`` before taking the basename (matching the
Mistral transcription fix and ``_normalize_filename``). This test is
cross-platform strict: the Windows-style ``.name`` contains backslashes
regardless of host, so a revert to splitting on ``/`` only fails on any OS.
"""

import io

from cognee.modules.ingestion.classify import classify


class _NamedRaw(io.RawIOBase):
    """A minimal raw stream exposing a controllable ``.name`` (proxied by
    ``BufferedReader.name``), so the test can force a Windows-style path on any
    host without touching the real filesystem."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        return 0


def _reader(name: str) -> io.BufferedReader:
    return io.BufferedReader(_NamedRaw(name))


def test_windows_path_name_becomes_basename():
    result = classify(_reader(r"C:\Users\me\audio\clip.mp3"))
    assert result.name == "clip.mp3"


def test_posix_path_name_becomes_basename():
    result = classify(_reader("/home/me/audio/clip.mp3"))
    assert result.name == "clip.mp3"


def test_explicit_filename_takes_precedence():
    # When a filename is passed explicitly, it is used verbatim and the
    # backslash path in ``.name`` is ignored.
    result = classify(_reader(r"C:\ignored\path.mp3"), filename="explicit.wav")
    assert result.name == "explicit.wav"
