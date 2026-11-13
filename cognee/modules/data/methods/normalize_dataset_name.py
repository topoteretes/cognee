import re

_FORBIDDEN_RUN = re.compile(r"[\s.]+")


def normalize_dataset_name(dataset_name: str) -> str:
    """Return the name cognee stores for a user-supplied dataset name.

    Dataset names cannot contain spaces or dots (see ``check_dataset_name``). A
    caller such as an agent naturally writes ``"software engineering"``; rather
    than rejecting it, every run of whitespace or dots becomes one underscore, so
    ``"software engineering"`` and ``"software_engineering"`` address the same
    dataset on every path that takes a name: add, cognify, search, delete.
    """
    return _FORBIDDEN_RUN.sub("_", dataset_name.strip())
