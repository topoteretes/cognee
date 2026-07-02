"""The one canonical key sanitizer the adapter core owns.

Every integration reuses this single function instead of re-implementing its
own, so ``dataset`` / ``session`` keys are derived identically everywhere. That
matters for two reasons:

* Keys are used verbatim as dataset names and session ids, which flow into
  storage paths, cache keys, and (for some backends) filesystem directories.
  A shared sanitizer keeps them collision-free and portable.
* It gives the conformance work a single source of truth to test against,
  rather than one subtly different implementation per bot.

The rules are intentionally boring and stable. Changing them would rewrite the
keys of already-stored memory, so treat this as a versioned contract.
"""

from __future__ import annotations

import re

# Any run of characters that is not a safe key character collapses to a single
# separator. "Safe" = lowercase alnum, and the structural characters we use to
# build keys (``:`` segment separator, ``-`` and ``_`` word separators).
_UNSAFE = re.compile(r"[^a-z0-9:_-]+")
# Collapse runs of the segment separator and trim stray separators at the edges
# of each segment so ``slack::#general`` and ``slack:#general:`` normalize alike.
_MULTI_COLON = re.compile(r":{2,}")

# Keep keys comfortably under backend identifier limits while staying unique.
_MAX_KEY_LEN = 255


def sanitize_token(token: str) -> str:
    """Normalize a single platform token (one key segment) to a safe form.

    Lowercases, replaces any unsafe run with ``-``, and trims separators from
    the ends. Returns ``""`` for input that is entirely unsafe/empty, letting
    the caller decide whether an empty segment is meaningful (e.g. a workspace
    that does not exist).

    Examples::

        sanitize_token("C123 General!")   -> "c123-general"
        sanitize_token("#général")         -> "g-n-ral"   # non-ascii dropped
        sanitize_token("")                  -> ""
    """
    if not token:
        return ""
    lowered = token.strip().lower()
    # Colons are the segment separator, not a within-token character.
    lowered = lowered.replace(":", "-")
    cleaned = _UNSAFE.sub("-", lowered)
    return cleaned.strip("-_")


def sanitize_key(*segments: str) -> str:
    """Join and sanitize segments into one canonical ``:``-separated key.

    Empty segments are dropped, so ``sanitize_key("chat", platform, workspace,
    channel)`` yields ``chat:slack:t1:c1`` and gracefully collapses to
    ``chat:slack:c1`` when ``workspace`` is empty. The result is lowercased,
    free of unsafe characters, and truncated to a safe length.

    Raises:
        ValueError: if every segment sanitizes away to nothing, since an empty
            key would silently alias unrelated conversations together.
    """
    parts = [sanitize_token(segment) for segment in segments]
    key = ":".join(part for part in parts if part)
    key = _MULTI_COLON.sub(":", key).strip(":")
    if not key:
        raise ValueError(f"Key sanitized to empty from segments: {segments!r}")
    if len(key) > _MAX_KEY_LEN:
        # Preserve a readable prefix and disambiguate with a short content hash
        # so distinct long keys never collide after truncation.
        import hashlib

        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        key = f"{key[: _MAX_KEY_LEN - 13]}:{digest}"
    return key
