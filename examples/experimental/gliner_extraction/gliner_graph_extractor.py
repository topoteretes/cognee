"""GLiNER2-backed graph extraction for cognee.

Builds a `calculate_chunk_graphs` callable that cognee's
`extract_graph_from_data` task accepts as a drop-in replacement for
LLM-based extraction. GLiNER2 (a 205M local encoder model) extracts
entities and relations from each chunk; results are mapped onto
cognee's `KnowledgeGraph` model, so the rest of the pipeline
(ontology resolution, node/edge construction, storage) is untouched.

Usage:
    from gliner2 import GLiNER2
    from gliner_graph_extractor import gliner_chunk_graph_calculator

    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    await cognee.cognify(
        datasets=["my_dataset"],
        calculate_chunk_graphs=gliner_chunk_graph_calculator(extractor),
    )
"""

import asyncio

from cognee.shared.data_models import KnowledgeGraph, Node, Edge

DEFAULT_ENTITY_TYPES = {
    "person": "Names of people or individuals",
    "organization": "Companies, institutions, or organizations",
    "location": "Geographic locations, cities, or places",
    "product": "Products, services, or technologies",
    "date": "Dates, years, or time references",
    "event": "Named events or occurrences",
}

DEFAULT_RELATION_TYPES = {
    "works_for": "Employment relationship where person works at organization",
    "founded": "Founding relationship where person created organization",
    "acquired": "Acquisition where one company bought another",
    "located_in": "Geographic relationship where entity is in a location",
    "part_of": "Membership or composition relationship",
    "created": "Creation relationship between entity and product or work",
}


def _span_text(value) -> str:
    """Normalize a GLiNER value: plain string, or a span dict when include_spans=True."""
    return value["text"] if isinstance(value, dict) else value


def _to_knowledge_graph(result: dict) -> KnowledgeGraph:
    """Map a GLiNER2 multi-task result onto cognee's KnowledgeGraph model."""
    nodes_by_id = {}
    edges = []
    seen_edges = set()

    for entity_type, values in result.get("entities", {}).items():
        for value in values:
            name = _span_text(value)
            if name not in nodes_by_id:
                nodes_by_id[name] = Node(
                    id=name,
                    name=name,
                    type=entity_type,
                    description=f"{name} is a {entity_type}.",
                )

    for relation, pairs in result.get("relation_extraction", {}).items():
        for pair in pairs:
            if isinstance(pair, dict):
                head, tail = _span_text(pair["head"]), _span_text(pair["tail"])
            else:
                head, tail = _span_text(pair[0]), _span_text(pair[1])
            if (head, relation, tail) in seen_edges:
                continue
            seen_edges.add((head, relation, tail))
            for endpoint in (head, tail):
                if endpoint not in nodes_by_id:
                    nodes_by_id[endpoint] = Node(
                        id=endpoint,
                        name=endpoint,
                        type="entity",
                        description=f"{endpoint} is an entity.",
                    )
            edges.append(
                Edge(
                    source_node_id=head,
                    target_node_id=tail,
                    relationship_name=relation,
                    description=f"{head} {relation.replace('_', ' ')} {tail}.",
                )
            )

    return KnowledgeGraph(nodes=list(nodes_by_id.values()), edges=edges)


def gliner_chunk_graph_calculator(
    extractor,
    entity_types: dict | list | None = None,
    relation_types: dict | list | None = None,
    threshold: float = 0.5,
    batch_size: int = 16,
):
    """Return a `calculate_chunk_graphs` callable for cognee's cognify.

    All chunks in a cognify batch (`chunks_per_batch`) are extracted in a
    single `batch_extract` call, so GLiNER encodes them in padded batches of
    `batch_size` instead of one forward pass per chunk (~3x faster on CPU).

    Args:
        extractor: A loaded gliner2 extractor (GLiNER2.from_pretrained(...)).
        entity_types: Entity labels, optionally with descriptions.
        relation_types: Relation labels, optionally with descriptions.
        threshold: Extraction confidence threshold.
        batch_size: Chunks per GLiNER forward pass.
    """
    entity_types = entity_types or DEFAULT_ENTITY_TYPES
    relation_types = relation_types or DEFAULT_RELATION_TYPES

    schema = extractor.create_schema().entities(entity_types).relations(relation_types)

    def _extract_batch_sync(texts: list[str]) -> list[KnowledgeGraph]:
        results = extractor.batch_extract(texts, schema, batch_size=batch_size, threshold=threshold)
        return [_to_knowledge_graph(result) for result in results]

    async def calculate_chunk_graphs(data_chunks, graph_model, custom_prompt=None, **kwargs):
        # GLiNER2 inference is synchronous CPU work — keep it off the event loop.
        return await asyncio.to_thread(_extract_batch_sync, [chunk.text for chunk in data_chunks])

    return calculate_chunk_graphs
