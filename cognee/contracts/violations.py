"""Contract violation handling.

Central logic for reacting to contract violations across all pipeline
checkpoints. Each violation mode (freeze, discard_row, discard_value, evolve)
has well-defined semantics matching DLT's contract system.
"""

from typing import Optional, List

from cognee.exceptions.exceptions import CogneeValidationError
from cognee.shared.logging_utils import get_logger

from .models import ContractMode, GraphContract

logger = get_logger("contracts")

# Sentinel returned by handle_violation for DISCARD_VALUE mode.
# Callers should check ``result is DISCARD_VALUE_SENTINEL`` and null the field.
DISCARD_VALUE_SENTINEL = object()


class ContractViolation(CogneeValidationError):
    """Raised when a contract rule is violated in FREEZE mode."""

    def __init__(
        self,
        message: str,
        *,
        contract_mode: ContractMode = ContractMode.FREEZE,
        entity_type: Optional[str] = None,
        entity_name: Optional[str] = None,
        data_item: object = None,
    ):
        self.contract_mode = contract_mode
        self.entity_type = entity_type
        self.entity_name = entity_name
        self.data_item = data_item
        super().__init__(
            message=message,
            name="ContractViolation",
        )


def handle_violation(
    mode: ContractMode,
    message: str,
    *,
    entity_type: Optional[str] = None,
    entity_name: Optional[str] = None,
    data_item: object = None,
):
    """Apply the appropriate reaction for a contract violation.

    Returns:
        - The original item for EVOLVE (pass-through).
        - None for DISCARD_ROW (caller should skip the item).
        - DISCARD_VALUE_SENTINEL for DISCARD_VALUE (caller should null the field).

    Raises:
        ContractViolation: In FREEZE mode.
    """
    if mode == ContractMode.FREEZE:
        raise ContractViolation(
            message,
            contract_mode=mode,
            entity_type=entity_type,
            entity_name=entity_name,
            data_item=data_item,
        )
    elif mode == ContractMode.DISCARD_ROW:
        logger.warning("Contract [discard_row]: %s", message)
        return None
    elif mode == ContractMode.DISCARD_VALUE:
        logger.warning("Contract [discard_value]: %s", message)
        return DISCARD_VALUE_SENTINEL
    else:
        # EVOLVE — log and pass through
        logger.info("Contract [evolve]: %s", message)
        return data_item


def apply_graph_contract(contract: GraphContract, chunk_graphs: list) -> list:
    """Filter nodes and edges in extracted KnowledgeGraphs according to a GraphContract.

    Mutates the graphs in-place and returns the list. Nodes/edges that violate
    the contract are handled per their contract mode (removed for discard_row,
    description cleared for discard_value, error for freeze, kept for evolve).
    """
    if contract is None:
        return chunk_graphs

    has_node_constraint = contract.allowed_node_types is not None
    has_edge_constraint = contract.allowed_edge_types is not None
    has_desc_constraint = contract.min_node_description_length > 0
    has_max_nodes = contract.max_nodes_per_chunk is not None

    # Fast path: no constraints configured
    if not (has_node_constraint or has_edge_constraint or has_desc_constraint or has_max_nodes):
        return chunk_graphs

    allowed_node_types_lower = (
        {t.lower() for t in contract.allowed_node_types} if has_node_constraint else None
    )
    allowed_edge_types_lower = (
        {t.lower() for t in contract.allowed_edge_types} if has_edge_constraint else None
    )

    for graph in chunk_graphs:
        nodes = getattr(graph, "nodes", None)
        edges = getattr(graph, "edges", None)
        if nodes is None or edges is None:
            continue

        # --- Node type enforcement ---
        if has_node_constraint:
            graph.nodes = _filter_nodes_by_type(
                nodes, allowed_node_types_lower, contract.node_types
            )

        # --- Node description length enforcement ---
        if has_desc_constraint:
            graph.nodes = _filter_nodes_by_description(
                graph.nodes, contract.min_node_description_length, contract.node_types
            )

        # --- Max nodes per chunk ---
        if has_max_nodes and len(graph.nodes) > contract.max_nodes_per_chunk:
            logger.warning(
                "Contract: chunk has %d nodes, capping to %d.",
                len(graph.nodes),
                contract.max_nodes_per_chunk,
            )
            graph.nodes = graph.nodes[: contract.max_nodes_per_chunk]

        # --- Edge type enforcement ---
        if has_edge_constraint:
            graph.edges = _filter_edges_by_type(
                edges, allowed_edge_types_lower, contract.edge_types
            )

        # --- Referential integrity: remove edges pointing to removed nodes ---
        valid_node_ids = {node.id for node in graph.nodes}
        graph.edges = [
            e
            for e in graph.edges
            if e.source_node_id in valid_node_ids and e.target_node_id in valid_node_ids
        ]

    return chunk_graphs


def _filter_nodes_by_type(
    nodes: list,
    allowed_types: set,
    mode: ContractMode,
) -> list:
    """Filter nodes whose type is not in the allowed set."""
    kept = []
    for node in nodes:
        node_type = getattr(node, "type", "")
        if node_type.lower() in allowed_types:
            kept.append(node)
        else:
            result = handle_violation(
                mode,
                f"Node type '{node_type}' (node '{node.name}') not in allowed types.",
                entity_type=node_type,
                entity_name=getattr(node, "name", None),
                data_item=node,
            )
            if result is not None and result is not DISCARD_VALUE_SENTINEL:
                # EVOLVE: keep the node as-is
                kept.append(node)
            elif result is DISCARD_VALUE_SENTINEL:
                # Clear the type but keep the node
                node.type = "unknown"
                kept.append(node)
            # DISCARD_ROW (None): node is dropped
    return kept


def _filter_nodes_by_description(
    nodes: list,
    min_length: int,
    mode: ContractMode,
) -> list:
    """Filter nodes whose description is too short."""
    kept = []
    for node in nodes:
        desc = getattr(node, "description", "") or ""
        if len(desc) >= min_length:
            kept.append(node)
        else:
            result = handle_violation(
                mode,
                f"Node '{node.name}' description too short ({len(desc)} < {min_length}).",
                entity_name=getattr(node, "name", None),
                data_item=node,
            )
            if result is not None and result is not DISCARD_VALUE_SENTINEL:
                kept.append(node)
            elif result is DISCARD_VALUE_SENTINEL:
                kept.append(node)
            # DISCARD_ROW: node dropped
    return kept


def _filter_edges_by_type(
    edges: list,
    allowed_types: set,
    mode: ContractMode,
) -> list:
    """Filter edges whose relationship_name is not in the allowed set."""
    kept = []
    for edge in edges:
        rel_name = getattr(edge, "relationship_name", "")
        if rel_name.lower() in allowed_types:
            kept.append(edge)
        else:
            result = handle_violation(
                mode,
                f"Edge type '{rel_name}' not in allowed types.",
                entity_type=rel_name,
                data_item=edge,
            )
            if result is not None and result is not DISCARD_VALUE_SENTINEL:
                kept.append(edge)
            elif result is DISCARD_VALUE_SENTINEL:
                edge.relationship_name = "related_to"
                kept.append(edge)
    return kept
