"""Minimal demo: GraphCompletion triplets -> Hybrid context + answer.

uv run python examples/demos/graph_completion_to_hybrid.py
"""

import asyncio
import pathlib
from typing import List, cast

import cognee
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import format_triplets

DATASET = "graph_completion_to_hybrid_demo"
QUERY = "Who works on Cognee, and how do Alice and Bob collaborate?"

DOCUMENTS = [
    "Alice is a Cognee engineer.",
    "Bob is Cognee's product manager.",
    "Cognee turns documents into AI memory.",
    "Alice builds Cognee hybrid retrieval.",
    "Alice and Bob meet weekly on Cognee demos.",
    "Bob sends Cognee feedback to Alice.",
]


def triplets_to_hybrid(edges: List[Edge]) -> dict:
    """Map GraphCompletion edges into Hybrid's chunks/ entities/ facts channels.

    Relationships already attached to an entity are not repeated as facts.
    ``made_from`` edges (summary↔chunk) are structural and skipped as facts.
    """
    chunks_by_id, entities_by_id, facts = {}, {}, []

    def label(n: Node) -> str:
        return n.attributes.get("name") or n.attributes.get("text") or n.id

    for edge in edges:
        for node in (edge.node1, edge.node2):
            ntype = node.attributes.get("type")
            text = node.attributes.get("text")
            if ntype in {"DocumentChunk", "TextSummary"} and text and node.id not in chunks_by_id:
                chunks_by_id[node.id] = {"id": node.id, "text": text}
            elif ntype == "Entity" and node.id not in entities_by_id:
                entities_by_id[node.id] = {
                    "id": node.id,
                    "name": node.attributes.get("name") or node.id,
                    "description": node.attributes.get("description"),
                    "edges": [],
                }

        rel = (
            edge.attributes.get("relationship_type")
            or edge.attributes.get("relationship_name")
            or edge.attributes.get("edge_text")
        )
        if not rel or rel == "made_from":
            continue

        bullet = f"{label(edge.node1)} -- {rel} -- {label(edge.node2)}"
        touches_entity = False
        for node in (edge.node1, edge.node2):
            entity = entities_by_id.get(node.id)
            if entity is not None:
                entity["edges"].append({"text": bullet})
                touches_entity = True

        # Already shown under an entity (or as passage/summary via chunks): skip facts.
        if not touches_entity:
            facts.append({"id": f"{edge.node1.id}:{edge.node2.id}", "text": bullet})

    return {
        "chunks": list(chunks_by_id.values()),
        "chunk_summaries": {},
        "entities": list(entities_by_id.values()),
        "facts": facts,
    }


async def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    cognee.config.system_root_directory(
        str(root / ".cognee_system/graph_completion_to_hybrid_demo")
    )
    cognee.config.data_root_directory(str(root / ".data_storage/graph_completion_to_hybrid_demo"))

    await cognee.forget(everything=True)
    await cognee.remember(DOCUMENTS, dataset_name=DATASET, self_improvement=False)

    retrieved = await GraphCompletionRetriever(top_k=8).get_retrieved_objects(query=QUERY)
    # get_retrieved_objects() is typed as Union[List[Edge], List[List[Edge]]] because it
    # also supports query_batch=. This demo only ever passes a single `query=`, so the
    # result is always a flat List[Edge] -- narrow it explicitly here (both for the type
    # checker and as a defensive runtime check) instead of accessing .node1/.node2
    # directly on a value the checker can't prove isn't a nested list.
    if retrieved and isinstance(retrieved[0], list):
        raise TypeError(
            "get_retrieved_objects() returned batch-mode results (List[List[Edge]]); "
            "this demo only supports the single-query List[Edge] shape."
        )
    edges = cast(List[Edge], retrieved)

    print("TRIPLETS\n", format_triplets(edges) if edges else "[none]")

    evidence = triplets_to_hybrid(edges)
    print(
        "\nCHANNEL COUNTS\n",
        f"chunks={len(evidence['chunks'])} "
        f"entities={len(evidence['entities'])} "
        f"facts={len(evidence['facts'])}",
    )

    hybrid = HybridRetriever()
    context = await hybrid.get_context_from_objects(query=QUERY, retrieved_objects=evidence)
    print("\nCONTEXT\n", context or "[empty]")

    answer = await hybrid.get_completion_from_context(
        query=QUERY, retrieved_objects=evidence, context=context
    )
    print("\nANSWER")
    for item in answer:
        print(item)


if __name__ == "__main__":
    asyncio.run(main())
