"""Contract model definitions.

Contracts control how Cognee reacts when data doesn't match expectations,
following the four-mode pattern from DLT: evolve, freeze, discard_row,
discard_value.
"""

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel


class ContractMode(str, Enum):
    """How to handle data that violates a contract rule.

    - EVOLVE: Accept and adapt (default). New types, columns, etc. are allowed.
    - FREEZE: Reject with a ContractViolation error.
    - DISCARD_ROW: Silently drop the entire item (node, edge, or data point).
    - DISCARD_VALUE: Null out the offending field but keep the item.
    """

    EVOLVE = "evolve"
    FREEZE = "freeze"
    DISCARD_ROW = "discard_row"
    DISCARD_VALUE = "discard_value"


class IngestionContract(BaseModel):
    """Controls what data enters the system during cognee.add().

    Enforced in ingest_data before data is stored in the relational DB.
    """

    max_file_size_bytes: Optional[int] = None
    allowed_mime_types: Optional[List[str]] = None
    allowed_extensions: Optional[List[str]] = None
    required_metadata_fields: Optional[List[str]] = None
    on_violation: ContractMode = ContractMode.FREEZE


class GraphContract(BaseModel):
    """Controls knowledge graph schema evolution during cognee.cognify().

    Enforced after LLM extraction, before graph integration.
    """

    node_types: ContractMode = ContractMode.EVOLVE
    edge_types: ContractMode = ContractMode.EVOLVE
    allowed_node_types: Optional[List[str]] = None
    allowed_edge_types: Optional[List[str]] = None
    min_node_description_length: int = 0
    max_nodes_per_chunk: Optional[int] = None


class DataContract(BaseModel):
    """Top-level contract combining all pipeline phases.

    Pass this to ``cognee.add(contract=...)`` or ``cognee.cognify(contract=...)``.
    Each sub-contract defaults to fully permissive (evolve mode).
    """

    ingestion: IngestionContract = IngestionContract()
    graph: GraphContract = GraphContract()
