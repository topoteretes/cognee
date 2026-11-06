"""A memory entry's code references reach the session cache and come back as given."""

import pytest
from pydantic import ValidationError

from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.exceptions.exceptions import CacheConnectionError
from cognee.memory import QAEntry, TraceEntry

# A real stamp: `enola locate --facts` over httpx at the named commit.
REF = {
    "repo": "httpx",
    "enola_id": "124022d9967b8736f8af972d89b79ab6",
    "name": "httpx/_client.AsyncClient.__aenter__",
    "file": "httpx/_client.py",
    "line": 1753,
    "body_hash": "015d90db5aea2353",
    "observed_commit": "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
    "extractor_version": "0.4.38",
}
STORED_REF = {**REF, "decl_hash": None, "resolution": "exact"}


def test_entries_validate_their_code_references():
    turn = QAEntry(question="q", answer="a", code_refs=[REF])
    assert turn.code_refs[0].resolution == "exact" and turn.code_refs[0].decl_hash is None

    step = TraceEntry(origin_function="read_file", code_refs=[{**REF, "resolution": "enclosing"}])
    assert step.code_refs[0].resolution == "enclosing"

    assert QAEntry(question="q", answer="a").code_refs is None

    with pytest.raises(ValidationError):
        QAEntry(question="q", answer="a", code_refs=[{**REF, "enola_id": "not-a-fact-id"}])
    with pytest.raises(ValidationError):
        QAEntry(question="q", answer="a", code_refs=[{**REF, "observed_commit": ""}])


@pytest.mark.asyncio
async def test_sql_cache_stores_code_references_as_given(tmp_path):
    cache = SqlCacheAdapter(f"sqlite+aiosqlite:///{tmp_path}/cache.db")
    try:
        await cache.create_qa_entry("u", "s", "q", "", "a", qa_id="qa-1", code_refs=[REF])
        await cache.create_qa_entry("u", "s", "q2", "", "a2", qa_id="qa-2")
        await cache.append_agent_trace_step(
            "u",
            "s",
            "t-1",
            "read_file",
            "success",
            code_refs=[REF, {**REF, "decl_hash": "f1f6c643329a9eb4", "resolution": "enclosing"}],
        )

        first, second = await cache.get_all_qa_entries("u", "s")
        assert first.code_refs == [STORED_REF]
        assert second.code_refs is None

        (step,) = await cache.get_agent_trace_session("u", "s")
        assert step.code_refs == [
            STORED_REF,
            {**REF, "decl_hash": "f1f6c643329a9eb4", "resolution": "enclosing"},
        ]

        with pytest.raises(CacheConnectionError):
            await cache.create_qa_entry(
                "u", "s", "q", "", "a", qa_id="qa-3", code_refs=[{"repo": "httpx"}]
            )
        assert len(await cache.get_all_qa_entries("u", "s")) == 2
    finally:
        await cache.close()
