"""The SearchType -> retriever table in cognee/modules/retrieval/README.md must match the code.

Source of truth is ``search_core_registry`` in
``get_search_type_retriever_instance.py`` plus the two types the factory handles
outside the dict (AGENTIC_COMPLETION, FEELING_LUCKY). Both files are read as text so
the check needs no database or LLM.
"""

import re
from pathlib import Path

from cognee.modules.search.types import SearchType

REPO = Path(__file__).resolve().parents[5]
README = REPO / "cognee" / "modules" / "retrieval" / "README.md"
FACTORY = (
    REPO / "cognee" / "modules" / "search" / "methods" / "get_search_type_retriever_instance.py"
)

# Handled by explicit branches in the factory rather than the registry dict.
OUT_OF_DICT = {
    "AGENTIC_COMPLETION": "AgenticRetriever",
    "FEELING_LUCKY": "—",
}


def _registry_from_source() -> dict[str, str]:
    src = FACTORY.read_text()
    body = src[src.index("search_core_registry: dict") :]
    pairs = re.findall(
        r"SearchType\.([A-Z_]+):\s*\(\s*(?:#[^\n]*\n\s*)*([A-Za-z0-9]+Retriever)", body
    )
    return dict(pairs)


def _table_from_readme() -> dict[str, str]:
    rows = {}
    for line in README.read_text().splitlines():
        m = re.match(r"^\|\s*`([A-Z_]+)`\s*\|\s*`?([^`|]+?)`?\s*\|", line)
        if m:
            rows[m.group(1)] = m.group(2).strip()
    return rows


def test_readme_table_matches_registry():
    expected = {**_registry_from_source(), **OUT_OF_DICT}
    actual = _table_from_readme()
    names = {m.name for m in SearchType}
    assert set(actual) == names, (
        "README table must list every SearchType exactly once: "
        f"missing={sorted(names - set(actual))} extra={sorted(set(actual) - names)}"
    )
    assert actual == expected, {
        k: (actual.get(k), expected.get(k))
        for k in set(actual) | set(expected)
        if actual.get(k) != expected.get(k)
    }


def test_registry_covers_every_search_type():
    covered = set(_registry_from_source()) | set(OUT_OF_DICT)
    assert covered == {m.name for m in SearchType}
