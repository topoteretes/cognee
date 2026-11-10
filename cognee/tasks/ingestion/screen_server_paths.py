"""Decide, once at the entry, what a caller's string is when server paths are refused.

With the server-path gate closed (a network caller, or ACCEPT_LOCAL_FILE_PATH=false),
a caller string is never looked up on this server's filesystem. Its shape alone
decides: a web URL or a database connection string goes on for the code that
handles it, a server path (a ``file://`` or ``s3://`` URL, or a string that is only
a path) is refused whether or not it exists, and anything else is marked literal
text (``DataItem.literal_text``), which every later step stores verbatim. No reply
reveals whether a path exists: with the gate closed, the later steps that would try
a string as a local file never ask the disk either. With the gate open nothing
changes.
"""

import re
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cognee.tasks.ingestion.create_dlt_source import is_connection_string
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.save_data_item_to_storage import (
    local_file_paths_accepted,
    require_server_path_access,
)


def screen_server_paths(data: Any) -> Any:
    """``data`` with every caller string decided by its shape when the gate is closed."""
    if local_file_paths_accepted():
        return data
    return _screen_item(data)


def _screen_item(item: Any) -> Any:
    """One input, lists and DataItems at any depth: a caller string decided, a
    ``Path`` refused, anything else (an upload, bytes) unchanged."""
    if isinstance(item, list):
        return [_screen_item(each) for each in item]
    if isinstance(item, DataItem):
        return _screen_data_item(item)
    if isinstance(item, Path):
        require_server_path_access()
    if isinstance(item, str) and _is_text(item):
        return DataItem(data=item, literal_text=True)
    return item


def _screen_data_item(item: DataItem) -> DataItem:
    """A caller's DataItem, its data decided like a bare input.

    ``cognee_path`` earns no exemption here: only ``resolve_data_directories`` sets
    it, after this runs, so on entry it is the caller's claim, not cognee's.
    """
    caller = replace(item, cognee_path=False)
    if isinstance(item.data, str):
        if item.literal_text or _is_text(item.data):
            return replace(caller, literal_text=True)
        return caller
    return replace(caller, data=_screen_item(item.data))


def _is_text(text: str) -> bool:
    """Whether a caller string is text; False for one that goes on as it is.

    Raises for a server path: the refusal never depends on whether it exists.
    """
    token = text.strip()
    parsed = urlparse(token)
    scheme = parsed.scheme.lower()
    if (scheme in ("http", "https") and parsed.netloc) or is_connection_string(text):
        return False  # a web page or repository to fetch, a database for its own gate
    # file:/x, file:///x, s3://bucket/key: one token, a "/" right after the colon.
    explicit = token[len(scheme) + 1 :].startswith("/") and not any(c.isspace() for c in token)
    if scheme in ("file", "s3") and explicit:
        require_server_path_access()
    if len(scheme) > 1 and parsed.netloc:
        # Another scheme (gs://, ftp://) is no server path. A one-letter scheme
        # is a Windows drive (C://Users/me) and falls through to the path rule.
        return True
    if is_path_only_string(text):
        require_server_path_access()
    return True


# How a path begins: a root or home (/, \, ~), ./ or ../, or a drive (C:\ or C:/).
_PATH_START = re.compile(r"[/\\~]|\.\.?[/\\]|[A-Za-z]:[/\\]")


def is_path_only_string(text: str) -> bool:
    """Whether ``text`` is nothing but a path, judged by its shape alone.

    One token after stripping (whitespace inside, newlines included, makes it
    prose) that is explicitly a path: it starts like one (a root, ``~``, ``./``,
    ``../`` or a drive) and holds a separator after its first character
    (``/etc/passwd``, ``~/notes``, ``./a``, ``C:\\data``), or it is a hidden
    file at the root (``/.env``). A string wrapped in one pair of matching
    quotes is unquoted first and may then hold spaces, though not line breaks:
    ``"C:\\Secret Folder"``. Anything less explicit is text, since it cannot be
    told from a word: ``/start``, ``~30%``, ``CI/CD``, ``12/25/2024``, a bare
    name such as ``notes.txt``, and a relative path such as ``docs/../x``.
    """
    token = text.strip()
    if len(token) > 1 and token[0] == token[-1] and token[0] in "\"'":
        token = token[1:-1]
        if any(char in "\r\n" for char in token):
            return False
    elif any(char.isspace() for char in token):
        return False
    if token.startswith("/."):
        return True
    return bool(_PATH_START.match(token)) and any(sep in token[1:] for sep in "/\\")
