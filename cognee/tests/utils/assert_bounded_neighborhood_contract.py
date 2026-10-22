from cognee.infrastructure.databases.graph.bounded_neighborhood import hop_distances


def assert_bounded_neighborhood_contract(
    chunks: list[tuple[list, list]],
    *,
    max_nodes: int,
    chunk_size: int,
    seed_ids: list[str],
    graph_edges: list[tuple] | None = None,
) -> tuple[list[str], set[tuple[str, str, str]]]:
    """Check chunks from ``iter_bounded_neighborhood`` against its contract.

    ``seed_ids`` are the seeds expected in the graph, in the order they must
    come first. With ``graph_edges`` (every edge of the test graph), member
    order is also checked to be non-decreasing in hop distance.

    Returns the member ids in order and the delivered edge identities.
    """
    members: list[str] = []
    delivered: set[str] = set()
    edge_identities: set[tuple[str, str, str]] = set()
    for chunk_nodes, chunk_edges in chunks:
        assert 0 < len(chunk_nodes) <= chunk_size, "a chunk holds 1..chunk_size nodes"
        for node_id, _ in chunk_nodes:
            node_key = str(node_id)
            assert node_key not in delivered, f"node {node_key} delivered twice"
            delivered.add(node_key)
            members.append(node_key)
        for edge in chunk_edges:
            identity = (str(edge[0]), str(edge[1]), str(edge[2]))
            assert identity not in edge_identities, f"edge {identity} delivered twice"
            assert identity[0] in delivered and identity[1] in delivered, (
                f"edge {identity} arrived before one of its endpoints"
            )
            edge_identities.add(identity)

    assert len(members) <= max_nodes
    assert members[: len(seed_ids)] == [str(seed_id) for seed_id in seed_ids][:max_nodes]

    if graph_edges is not None:
        distance = hop_distances(graph_edges, seed_ids)
        hops = [distance[node_id] for node_id in members]
        assert hops == sorted(hops), "members must come nearest hop first"
    return members, edge_identities
