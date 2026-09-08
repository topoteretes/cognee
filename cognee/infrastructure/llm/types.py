"""Framework-neutral LLM return types shared by every structured-output framework."""

from typing import Any


class TranscriptionReturnType:
    """Audio transcription result.

    ``text`` is the transcript; ``payload`` is the provider's raw response (e.g. a
    litellm ``TranscriptionResponse``) so callers can read extras such as
    ``verbose_json`` segments without depending on the transport used.
    """

    text: str
    payload: Any

    def __init__(self, text: str, payload: Any):
        self.text = text
        self.payload = payload
