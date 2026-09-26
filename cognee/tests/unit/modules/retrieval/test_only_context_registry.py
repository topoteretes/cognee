"""Every SearchType has an explicit ``only_context`` contract: full LLM input or bare context.

The retriever for each type is built through the real factory and checked against the
pinned table below. A new SearchType fails here until someone decides which side it is
on, and a retriever that gains prompt-template attributes without sending exactly one
templated prompt at completion time must opt out with ``supports_prompt_preview = False``
(as Cypher and the agentic loop do) or land on the bare-context side here.
"""

import types
from uuid import uuid4

import pytest

from cognee.modules.retrieval.only_context_prompt import retriever_sends_one_prompt
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.types import SearchType

# Completion types whose get_completion_from_context sends exactly one prompt rendered from
# the retriever's own (user_prompt_path, system_prompt_path) pair.
RETURNS_FULL_PROMPT = {
    SearchType.GRAPH_COMPLETION,
    SearchType.GRAPH_SUMMARY_COMPLETION,
    SearchType.GRAPH_COMPLETION_COT,
    SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
    SearchType.GRAPH_COMPLETION_DECOMPOSITION,
    SearchType.HYBRID_COMPLETION,
    SearchType.RAG_COMPLETION,
    SearchType.TRIPLET_COMPLETION,
    SearchType.TEMPORAL,
    # Reading and table queries happen during retrieval; the answer is one prompt from the
    # pair, over the count code computed.
    SearchType.BROAD,
}

# Non-generative types (no template at all) and the opt-outs (templates present, but the
# answer never comes from that single pair).
RETURNS_BARE_CONTEXT = {
    SearchType.CHUNKS,
    SearchType.CHUNKS_LEXICAL,
    SearchType.SUMMARIES,
    SearchType.CYPHER,
    SearchType.NATURAL_LANGUAGE,
    SearchType.CODING_RULES,
    SearchType.SKILLS,
    SearchType.AGENTIC_COMPLETION,
    SearchType.CODE,
    SearchType.GRAPH_REPORT,
}

# FEELING_LUCKY resolves to one of the others before a retriever is built.
CONSTRUCTIBLE = sorted(set(SearchType) - {SearchType.FEELING_LUCKY}, key=lambda s: s.name)


def test_every_search_type_has_a_pinned_only_context_contract():
    assert RETURNS_FULL_PROMPT.isdisjoint(RETURNS_BARE_CONTEXT)
    assert RETURNS_FULL_PROMPT | RETURNS_BARE_CONTEXT == set(CONSTRUCTIBLE), (
        "new SearchType: decide whether only_context returns the full LLM input or the bare"
        " context, and add it to the matching table above"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", CONSTRUCTIBLE, ids=lambda s: s.name)
async def test_registered_retriever_matches_the_pinned_contract(search_type):
    retriever = await get_search_type_retriever_instance(
        search_type,
        query_text="q",
        user=types.SimpleNamespace(id=uuid4(), tenant_id=None),
        dataset=types.SimpleNamespace(id=uuid4(), name="ds", tenant_id=None),
    )

    assert retriever_sends_one_prompt(retriever) is (search_type in RETURNS_FULL_PROMPT), (
        f"{search_type.name}: {type(retriever).__name__} is on the wrong side of the table"
    )
