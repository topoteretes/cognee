"""Deduplication utilities for nodes and edges in the knowledge graph.

This module provides functions to deduplicate nodes and edges before they are
added to the graph database. Understanding the deduplication behavior is crucial
for managing entity property conflicts and updates.

Deduplication Strategy: First Write Wins (ID-based Skip)
=========================================================

The current implementation uses a **First Write Wins** strategy with pure ID-based
filtering. This means:

1. **For Nodes**: If a node with the same ID already exists in the batch, subsequent
   nodes with that ID are completely skipped - ALL their properties are ignored.

2. **For Edges**: If an edge with the same key (source_id + relationship_type + target_id)
   already exists, subsequent edges with that key are skipped.

Important Implications
----------------------

**Missing Properties Are NOT Added:**
    When you ingest new data with additional properties for an existing entity,
    those new properties will be lost. The entity is identified by ID and skipped
    entirely.

    Example::

        # First ingestion
        await cognee.add({"id": "product-1", "name": "Backpack", "weight": 500})
        await cognee.cognify()

        # Second ingestion - adds new property
        await cognee.add({"id": "product-1", "name": "Backpack", "capacity": 40})
        await cognee.cognify(incremental_loading=True)

        # Result: 'capacity' property is LOST
        # The node is completely skipped because str(node.id) already exists

**Property Conflicts Are Ignored (First Value Preserved):**
    When you ingest data with different values for existing properties,
    the new values are ignored - the original values are preserved.

    Example::

        # First ingestion
        await cognee.add({"id": "product-1", "weight": 500})
        await cognee.cognify()

        # Second ingestion - different value
        await cognee.add({"id": "product-1", "weight": 520})
        await cognee.cognify(incremental_loading=True)

        # Result: weight remains 500
        # New value (520) is completely ignored

Currently NOT Supported
-----------------------

- Merging properties from multiple ingestions
- Updating existing property values
- Adding missing properties to existing entities
- Configurable merge strategies
- Property history or versioning

Workarounds
-----------

1. **Include all properties in every ingestion** - partial updates will lose data
2. **Prune and re-ingest** to make changes: ``await cognee.prune.prune_data()``
3. **Implement external merge logic** before calling ``cognee.add()``
4. **Maintain a separate source of truth** database and regenerate the full graph
5. **Use direct Cypher queries** if using Memgraph/Neo4j, bypassing Cognee's ingestion

See Also
--------
GitHub Issue #1831 for feature requests regarding configurable merge strategies.
"""

from cognee.infrastructure.engine import DataPoint


def deduplicate_nodes_and_edges(nodes: list[DataPoint], edges: list[dict]):
    """Remove duplicate nodes and edges from the input lists.

    This function implements a First Write Wins deduplication strategy.
    Nodes and edges are identified by their IDs, and only the first occurrence
    of each unique ID is kept. Subsequent occurrences are completely discarded,
    including all their properties.

    Parameters
    ----------
    nodes : list[DataPoint]
        List of DataPoint objects representing nodes to be added to the graph.
        Each node must have an 'id' attribute that uniquely identifies it.
    edges : list[dict]
        List of edge dictionaries. Each edge is expected to be a tuple-like
        structure where edge[0] is the source ID, edge[1] is the relationship
        type, and edge[2] is the target ID.

    Returns
    -------
    tuple[list[DataPoint], list[dict]]
        A tuple containing:
        - final_nodes: Deduplicated list of nodes (first occurrence of each ID)
        - final_edges: Deduplicated list of edges (first occurrence of each key)

    Notes
    -----
    **Deduplication Behavior:**

    - Nodes are deduplicated based on ``str(node.id)``
    - Edges are deduplicated based on a composite key: ``str(edge[0]) + str(edge[2]) + str(edge[1])``
      (source_id + target_id + relationship_type)
    - Only the FIRST occurrence is kept; all subsequent duplicates are discarded
    - No property merging occurs - duplicate entities are completely ignored

    **Important Limitations:**

    This function does NOT:
    - Merge properties from duplicate nodes
    - Update existing property values
    - Add missing properties to existing entities
    - Provide any conflict resolution beyond "first wins"

    For use cases requiring property merging or updates, consider:
    - Pre-processing data before calling this function
    - Using direct database queries with MERGE operations
    - Implementing custom deduplication logic

    Examples
    --------
    >>> from cognee.infrastructure.engine import DataPoint
    >>> nodes = [
    ...     DataPoint(id="1", name="First", value=100),
    ...     DataPoint(id="1", name="Duplicate", value=200),  # Will be skipped
    ...     DataPoint(id="2", name="Second", value=300),
    ... ]
    >>> edges = []
    >>> deduped_nodes, deduped_edges = deduplicate_nodes_and_edges(nodes, edges)
    >>> len(deduped_nodes)
    2
    >>> deduped_nodes[0].value  # First occurrence kept
    100
    """
    added_entities = {}
    final_nodes = []
    final_edges = []

    for node in nodes:
        if str(node.id) not in added_entities:
            final_nodes.append(node)
            added_entities[str(node.id)] = True

    for edge in edges:
        edge_key = str(edge[0]) + str(edge[2]) + str(edge[1])
        if edge_key not in added_entities:
            final_edges.append(edge)
            added_entities[edge_key] = True

    return final_nodes, final_edges
