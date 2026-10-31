"""Minimal demo: GraphCompletion triplets -> Hybrid context + answer.

uv run python examples/demos/graph_completion_to_hybrid.py
"""

import asyncio
import pathlib
from typing import cast

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


def node_label(node: Node) -> str:
    return node.attributes.get("name") or node.attributes.get("text") or node.id


def create_chunk_entry(node: Node) -> dict:
    return {"id": node.id, "text": node.attributes.get("text")}


def create_entity_entry(node: Node) -> dict:
    return {
        "id": node.id,
        "name": node.attributes.get("name") or node.id,
        "description": node.attributes.get("description"),
        "type": node.attributes.get("type"),
        "edges": [],
    }


def create_entity_edge_entry(edge: Edge) -> dict:
    relationship = edge.attributes.get("relationship_name") or edge.attributes.get(
        "relationship_type"
    )
    text = edge.attributes.get("edge_text") or (
        f"{node_label(edge.node1)} -- {relationship} -- {node_label(edge.node2)}"
    )
    return {
        "text": text,
        "source": node_label(edge.node1),
        "target": node_label(edge.node2),
        "source_id": edge.node1.id,
        "relationship": relationship,
        "target_id": edge.node2.id,
        "edge_object_id": edge.attributes.get("edge_object_id"),
    }


def create_fact_entry(edge: Edge) -> dict:
    return {
        "id": f"{edge.node1.id}:{edge.node2.id}",
        "text": create_entity_edge_entry(edge)["text"],
    }


def triplets_to_hybrid(edges: list[Edge]) -> dict:
    """Map GraphCompletion edges into Hybrid's chunks / entities / facts channels.

    ``made_from`` edges pair a summary with its source chunk (GraphCompletion
    does not project ``TextSummary.source_chunk_id``). Unpaired summaries are
    kept as passages. Edges that already hang off an entity are not repeated
    as facts.
    """
    chunks, summaries, entities, facts = {}, {}, {}, []
    summary_to_chunk = {}

    for edge in edges:
        for node in (edge.node1, edge.node2):
            ntype = node.attributes.get("type")
            text = node.attributes.get("text")
            if ntype == "DocumentChunk" and text:
                chunks.setdefault(node.id, create_chunk_entry(node))
            elif ntype == "TextSummary" and text:
                summaries.setdefault(node.id, create_chunk_entry(node))
            elif ntype == "Entity":
                entities.setdefault(node.id, create_entity_entry(node))

        relationship = edge.attributes.get("relationship_name")
        types = (edge.node1.attributes.get("type"), edge.node2.attributes.get("type"))
        if relationship == "made_from":
            if types == ("TextSummary", "DocumentChunk"):
                summary_to_chunk[edge.node1.id] = edge.node2.id
            elif types == ("DocumentChunk", "TextSummary"):
                summary_to_chunk[edge.node2.id] = edge.node1.id
            continue

        if not relationship and not edge.attributes.get("edge_text"):
            continue

        edge_entry = create_entity_edge_entry(edge)
        attached = False
        for node in (edge.node1, edge.node2):
            if node.id not in entities:
                continue
            entities[node.id]["edges"].append(edge_entry)
            attached = True
        if not attached:
            facts.append(create_fact_entry(edge))

    chunk_summaries = {}
    for summary_id, summary in summaries.items():
        chunk_id = summary_to_chunk.get(summary_id)
        if chunk_id in chunks:
            chunk_summaries[chunk_id] = summary["text"]
        else:
            chunks.setdefault(summary_id, summary)

    return {
        "chunks": list(chunks.values()),
        "chunk_summaries": chunk_summaries,
        "entities": list(entities.values()),
        "facts": facts,
    }


async def main() -> None:
    demo_dir = pathlib.Path(__file__).parent / ".cognee_system"
    cognee.config.system_root_directory(str(demo_dir))
    cognee.config.data_root_directory(str(demo_dir / "data"))

    await cognee.forget(everything=True)
    await cognee.remember(DOCUMENTS, dataset_name=DATASET, self_improvement=False)

    edges = cast(
        list[Edge],
        await GraphCompletionRetriever(top_k=8).get_retrieved_objects(query=QUERY),
    )

    print("TRIPLETS\n", format_triplets(edges) if edges else "[none]")

    evidence = triplets_to_hybrid(edges)
    print(
        "\nCHANNEL COUNTS\n",
        f"chunks={len(evidence['chunks'])} "
        f"chunk_summaries={len(evidence['chunk_summaries'])} "
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
