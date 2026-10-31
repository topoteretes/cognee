"""Bounded, approximate seed selection for Cypher graph adapters.

The sample is a physical prefix, not a random or temporally representative
sample. New hubs beyond it can remain invisible as ingestion grows the graph.
It bounds aggregation and returned data, not the engine's query-planning cost.
"""

EDGE_SAMPLE_ROWS = 200_000


async def cypher_degree_seeds(adapter, top_k: int, *, typed: bool = False) -> list[str]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    node_label = ":Node" if typed else ""
    edge_label = ":EDGE" if typed else ""
    rows = await adapter.query(
        f"""
        MATCH (source{node_label})-[edge{edge_label}]->(target{node_label})
        WITH source.id AS source_id, target.id AS target_id
        LIMIT $sample
        UNWIND [source_id, target_id] AS id
        WITH id WHERE id IS NOT NULL
        RETURN id, count(*) AS degree
        ORDER BY degree DESC, id
        LIMIT $top_k
        """,
        {"sample": EDGE_SAMPLE_ROWS, "top_k": top_k},
    )

    def ids(results):
        return [str(row["id"] if isinstance(row, dict) else row[0]) for row in results]

    seeds = ids(rows)
    if len(seeds) < top_k:
        # Fill edgeless/sparse samples using ids only, never full graph objects.
        rows = await adapter.query(
            f"MATCH (n{node_label}) WHERE n.id IS NOT NULL AND NOT n.id IN $seed_ids "
            "RETURN n.id AS id LIMIT $remaining",
            {"seed_ids": seeds, "remaining": top_k - len(seeds)},
        )
        seeds.extend(ids(rows))
    return seeds
