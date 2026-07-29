"""Unit tests for LangMemSource.

Pure: no databases, no LLM calls, no network. Exercises the dump parsing and
COGX record mapping for the shapes a LangMem export can take.
"""

import json
from pathlib import Path

import pytest

from cognee.modules.migration.cogx import COGXMemory
from cognee.modules.migration.sources.langmem import LangMemSource


def _collect(source) -> list:
    """Drain the async records() generator into a list."""
    import asyncio

    return asyncio.run(_drain(source))


async def _drain(source) -> list:
    return [record async for record in source.records()]


_SAMPLE = [
    {
        "id": "lm-1",
        "content": "User prefers tea over coffee.",
        "categories": ["preference"],
        "user_id": "u1",
        "created_at": "2024-01-02T03:04:05Z",
        "metadata": {"source": "chat"},
    },
    {
        "id": "lm-2",
        "text": "Project Apollo launches in Q3.",
        "namespace": "team-a",
        "createdAt": "2024-02-03T04:05:06+00:00",
    },
]


def test_list_of_dicts_yields_memories():
    source = LangMemSource(_SAMPLE, mode="preserve")
    records = _collect(source)

    assert len(records) == 2
    assert all(isinstance(r, COGXMemory) for r in records)
    assert records[0].content == "User prefers tea over coffee."
    assert records[0].external_id == "lm-1"
    assert records[0].categories == ["preference"]
    assert records[0].scope.user_id == "u1"
    assert records[0].metadata == {"langmem_metadata": {"source": "chat"}}
    assert records[0].created_at is not None
    assert records[0].created_at.year == 2024
    # ``namespace`` falls back to ``user_id`` for the scope.
    assert records[1].scope.user_id == "team-a"
    assert records[1].content == "Project Apollo launches in Q3."
    assert records[1].created_at is not None


def test_dict_wrapper_is_unwrapped():
    source = LangMemSource({"memories": _SAMPLE}, mode="re-derive")
    records = _collect(source)

    assert len(records) == 2
    assert records[0].external_id == "lm-1"


def test_file_path_is_read():
    path = Path(__file__).parent / "langmem_sample_dump.json"
    path.write_text(json.dumps(_SAMPLE), encoding="utf-8")
    try:
        source = LangMemSource(str(path), mode="hybrid")
        records = _collect(source)
        assert len(records) == 2
        assert records[0].external_id == "lm-1"
    finally:
        path.unlink()


def test_non_string_content_and_non_dicts_are_skipped():
    dump = [
        {"id": "ok", "content": "valid memory"},
        {"id": "bad", "data": 123},  # no string content
        "not-a-dict",
        {"id": "also-ok", "text": "another memory"},
    ]
    source = LangMemSource(dump)
    records = _collect(source)

    assert [r.external_id for r in records] == ["ok", "also-ok"]


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        _collect(LangMemSource({"unexpected": "shape"}))


def test_missing_content_yields_nothing():
    source = LangMemSource([{"id": "x", "meta": {}}])
    assert _collect(source) == []


def test_mode_does_not_affect_records():
    for mode in ("re-derive", "preserve", "hybrid"):
        source = LangMemSource(_SAMPLE, mode=mode)
        assert len(_collect(source)) == 2


def test_generated_external_id_when_missing():
    source = LangMemSource([{"content": "orphan memory without id"}])
    records = _collect(source)

    assert len(records) == 1
    assert records[0].external_id == "langmem-0"
