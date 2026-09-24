"""Shared pieces of ``GraphDBInterface.iter_bounded_neighborhood``.

The chunking rule lives here so the interface default and every native adapter
hand chunks to callers under the same contract: nodes in membership order, and
each edge between two members exactly once, in the chunk holding its later
endpoint. A caller can therefore draw every chunk as it arrives.
"""

from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any

DEFAULT_NEIGHBORHOOD_CHUNK_SIZE = 2000


def validate_bounded_neighborhood_args(depth: int, max_nodes: int, chunk_size: int) -> None:
    """Reject meaningless bounds before an adapter reads anything."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")


def unique_node_ids(node_ids: Iterable[Any]) -> list[str]:
    """Stringified node ids, de-duplicated in first-seen order."""
    return list(dict.fromkeys(str(node_id) for node_id in node_ids))


def hop_distances(edges: Iterable[tuple], seed_ids: Iterable[Any]) -> dict[str, int]:
    """Undirected breadth-first hop distance from the seeds over ``edges``."""
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        source_key, target_key = str(edge[0]), str(edge[1])
        adjacency.setdefault(source_key, set()).add(target_key)
        adjacency.setdefault(target_key, set()).add(source_key)

    distance: dict[str, int] = {}
    queue: deque = deque()
    for seed_id in seed_ids:
        seed_key = str(seed_id)
        if seed_key not in distance:
            distance[seed_key] = 0
            queue.append(seed_key)
    while queue:
        node_id = queue.popleft()
        for neighbor_id in adjacency.get(node_id, ()):
            if neighbor_id not in distance:
                distance[neighbor_id] = distance[node_id] + 1
                queue.append(neighbor_id)
    return distance


def order_members(
    nodes: list[tuple[str, dict]],
    edges: list[tuple],
    seed_ids: list[str],
    max_nodes: int,
) -> list[tuple[str, dict]]:
    """The first ``max_nodes`` nodes: seeds in seed order, then by hop distance.

    Nodes at the same distance keep the order they came in, which is the
    adapter's. Seeds absent from ``nodes`` are not in the graph and take no slot.
    """
    distance = hop_distances(edges, seed_ids)
    seed_rank = {seed_id: index for index, seed_id in enumerate(seed_ids)}
    # A node listed twice is one member; it keeps its first position.
    unique_nodes = list({str(node_id): (node_id, data) for node_id, data in nodes}.values())
    position = {str(node_id): index for index, (node_id, _) in enumerate(unique_nodes)}

    def rank(node: tuple[str, dict]) -> tuple[int, int]:
        node_key = str(node[0])
        hops = distance.get(node_key, 10_000)
        return hops, seed_rank[node_key] if hops == 0 else position[node_key]

    return sorted(unique_nodes, key=rank)[:max_nodes]


def project_properties(properties: dict, property_keys: list[str] | None) -> dict:
    """``properties`` cut down to ``name``, ``type`` and ``property_keys``.

    ``None`` keeps everything. Keys the node does not have are not invented.
    """
    if property_keys is None:
        return properties
    wanted = ("name", "type", *property_keys)
    return {key: properties[key] for key in wanted if key in properties}


def chunk_members(
    members: list[tuple[str, dict]],
    edges: Iterable[tuple],
    chunk_size: int,
    property_keys: list[str] | None = None,
) -> Iterator[tuple[list[tuple[str, dict]], list[tuple]]]:
    """Split members into chunks under the ``iter_bounded_neighborhood`` rules.

    Edges with an endpoint outside ``members`` are dropped. Every other edge is
    placed in the chunk of its later endpoint, so it is yielded exactly once and
    never before both of its endpoints.
    """
    position = {str(node_id): index for index, (node_id, _) in enumerate(members)}
    edges_by_chunk: dict[int, list[tuple]] = {}
    for edge in edges:
        source_position = position.get(str(edge[0]))
        target_position = position.get(str(edge[1]))
        if source_position is None or target_position is None:
            continue
        chunk_index = max(source_position, target_position) // chunk_size
        edges_by_chunk.setdefault(chunk_index, []).append(edge)

    for chunk_index, start in enumerate(range(0, len(members), chunk_size)):
        chunk_nodes = [
            (node_id, project_properties(data, property_keys))
            for node_id, data in members[start : start + chunk_size]
        ]
        yield chunk_nodes, edges_by_chunk.get(chunk_index, [])
