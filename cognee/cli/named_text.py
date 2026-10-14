"""``--name`` for inline text on ``cognee add`` and ``cognee remember``.

Text typed on the command line has no filename, so cognee names it by its
content (``text_<md5>``, or ``text_<n>`` on the API lane). ``--name`` gives
that one document a name instead. It applies to inline text only: a file
keeps its own basename, so a name next to several texts would be ambiguous,
and the option is refused unless exactly one item is inline text.
"""

import os
import re
from pathlib import PurePath

NAME_HELP = (
    "Document name for text given inline (e.g. --name 'Q3 plan'); without it the "
    "text is stored under a content-hash name. Applies to exactly one text item; "
    "files keep their own filename."
)


_URL_LIKE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def inline_text_indices(data_items: list[str]) -> list[int]:
    """Positions of the items that are inline text: not a file, directory or URL.

    A ``file://``, ``s3://`` or ``http(s)://`` item is resolved by cognee, not
    stored as text, so a name could not apply to it.
    """
    return [
        index
        for index, item in enumerate(data_items)
        if not os.path.exists(item) and not _URL_LIKE.match(item)
    ]


def check_name(data_items: list[str], name: str | None) -> None:
    """Refuse ``--name`` unless it can only mean one thing."""
    if name is None:
        return
    if not name.strip():
        raise ValueError("--name must not be empty.")
    texts = inline_text_indices(data_items)
    if len(texts) != 1:
        raise ValueError(
            "--name applies to exactly one inline text item; "
            f"got {len(texts)} (files keep their own filename)."
        )


def upload_filename(name: str) -> str:
    """The multipart filename for a named text: its basename, ending in ``.txt``."""
    base = PurePath(name.strip().replace("\\", "/")).name.strip() or "text"
    return base if base.lower().endswith(".txt") else f"{base}.txt"


def name_inline_text(data_items: list[str], name: str | None) -> list:
    """The items for the in-process lane: the named text becomes a ``DataItem``.

    Everything else (files, URLs, unnamed text) is passed through unchanged, so
    cognee resolves it exactly as before. ``check_name`` has already ensured
    the name can only refer to one item.
    """
    if name is None:
        return list(data_items)
    from cognee.tasks.ingestion.data_item import DataItem

    (index,) = inline_text_indices(data_items)
    items: list = list(data_items)
    items[index] = DataItem(data=data_items[index], name=name.strip())
    return items
