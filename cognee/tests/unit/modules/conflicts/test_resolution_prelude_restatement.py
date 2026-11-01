"""A user decision is stated twice when the global-context prelude is on (SDK-1328).

Scenario. The user resolved the conflict "who is Company's CEO" by accepting the
Alice fact over the Bob fact. A later GRAPH_COMPLETION or HYBRID search on a
retriever with the global-context prelude enabled retrieves both edges.

What happens on the branch:

1. ``apply_graph_resolutions`` drops the rejected Bob edge, because the selected
   Alice edge was retrieved too.
2. ``resolve_edges_to_text`` renders the Alice edge with the ``[accepted by user]``
   label and, since the selected edge is present, restates nothing itself.
3. The retriever then builds the prelude from ``accepted_facts_context`` with the
   ids of the nodes in context. The filter is "any retrieved node is the decision's
   subject, a value or a source", and the Company node, the subject of the labelled
   Alice edge, is always in context. So the prelude adds
   "Accepted by user: Company has CEO Alice; ignore competing claims."

The same decision reaches the LLM twice: once as the label on the edge, once in
the "## Accepted facts (user resolved)" section. The test fails until the prelude
skips decisions whose selected edge is already in the context.
"""

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text
from cognee.modules.retrieval.utils.resolved_context import (
    accepted_facts_context,
    apply_graph_resolutions,
    user_resolution_text,
)
from cognee.modules.retrieval.utils.used_graph_elements import extract_from_edges

DECISION = {
    "id": "conflict",
    "about_id": "company",
    "selected_fact": "Company has CEO Alice",
    "selected_fact_id": "alice-fact",
    "rejected_fact_ids": ["bob-fact"],
    "sources": [],
    "values": [],
}
USER_MARK = {
    "conflict_id": "conflict",
    "resolution_source": "user",
    "selected_fact": "Company has CEO Alice",
}


def _retrieved_edges() -> list[Edge]:
    company = Node("company", {"name": "Company"})
    alice = Edge(
        company,
        Node("alice", {"name": "Alice"}),
        attributes={
            "edge_object_id": "alice-fact",
            "edge_text": "Company has CEO Alice",
            "conflict_marks": [{**USER_MARK, "status": "current"}],
        },
    )
    bob = Edge(
        company,
        Node("bob", {"name": "Bob"}),
        attributes={
            "edge_object_id": "bob-fact",
            "edge_text": "Company has CEO Bob",
            "conflict_marks": [{**USER_MARK, "status": "superseded"}],
        },
    )
    return [alice, bob]


@pytest.mark.asyncio
async def test_a_user_decision_is_stated_once_with_the_global_context_prelude():
    edges = apply_graph_resolutions(_retrieved_edges(), [DECISION])
    graph_context = await resolve_edges_to_text(edges)

    # What GraphCompletionRetriever / HybridRetriever hand to accepted_facts_context:
    # the ids of the nodes and edges that ended up in the context.
    used = extract_from_edges(edges)
    prelude = accepted_facts_context([DECISION], used["node_ids"], used.get("edge_ids", []))

    context = "\n\n".join(section for section in (prelude, graph_context) if section)
    labelled = context.count("[accepted by user]")
    restated = context.count(user_resolution_text("Company has CEO Alice"))
    assert labelled + restated == 1, (
        f"decision stated {labelled + restated} times "
        f"({labelled} as the edge label, {restated} in the prelude):\n{context}"
    )
