"""Worker endpoints as they may appear in logs, spans and errors."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def redact_url(url: str) -> str:
    """``scheme://user:pass@host`` -> ``scheme://***@host``; each comma-separated part."""
    parts = []
    for part in url.split(","):
        part = part.strip()
        try:
            split = urlsplit(part)
        except ValueError:
            parts.append("<invalid url>")
            continue
        if "@" in split.netloc:
            # The last "@" ends the userinfo, so a password containing "@" stays hidden.
            host = split.netloc.rsplit("@", 1)[1]
            part = urlunsplit(split._replace(netloc=f"***@{host}"))
        parts.append(part)
    return ",".join(parts)
