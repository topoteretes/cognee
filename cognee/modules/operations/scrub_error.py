"""Bounded, PII-scrubbed error messages for pipeline_runs (SDK-399 follow-up).

Exception messages routinely echo user data (emails, file paths under a
home directory, API keys from provider errors, long identifiers). The
``error_message`` column is meant for operators debugging failures, so the
text is redacted with conservative patterns and hard-truncated before it is
persisted. The raw message still reaches logs/tracebacks as before — only
the durable typed column is scrubbed.

This module must stay import-light (stdlib only).
"""

import re
from typing import Optional

ERROR_MESSAGE_MAX_LENGTH = 512

# Order matters: secret-shaped tokens are redacted before digit runs so a
# key is reported as one [secret], not a shredded mix of both markers.
_SCRUB_PATTERNS = [
    # email addresses
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    # bearer tokens and key-prefixed credentials (sk-..., api_key_..., token-...)
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"), "[secret]"),
    (re.compile(r"(?i)\b(?:sk|pk|rk|api|key|token|secret)[-_][A-Za-z0-9._-]{8,}\b"), "[secret]"),
    # long hex/base64-ish blobs (hashes, keys, signed payloads)
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "[secret]"),
    (re.compile(r"\b[A-Za-z0-9+/=_-]{40,}\b"), "[secret]"),
    # home-directory user names in paths
    (re.compile(r"(/(?:Users|home)/)[^/\s]+"), r"\1[user]"),
    (re.compile(r"([A-Za-z]:\\Users\\)[^\\\s]+"), r"\1[user]"),
    # long digit runs (phone numbers, account/card numbers)
    (re.compile(r"\b\d{6,}\b"), "[number]"),
]


def scrub_error_message(message: Optional[object]) -> Optional[str]:
    """Redact PII/secret-shaped substrings and truncate to the column bound."""
    if message is None:
        return None
    text = str(message)
    if not text:
        return None
    for pattern, replacement in _SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > ERROR_MESSAGE_MAX_LENGTH:
        text = text[: ERROR_MESSAGE_MAX_LENGTH - 1] + "…"
    return text
