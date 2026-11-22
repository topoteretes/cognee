"""Whether a string is nothing but a server path, judged by its shape alone.

No disk lookup and no cognee imports: the entry screen (``screen_server_paths``)
and the storage steps it protects both refuse by this rule, without an import
cycle between them.
"""

import re

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
