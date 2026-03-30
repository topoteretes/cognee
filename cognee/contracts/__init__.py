"""Data contracts for controlling schema evolution and data validation in Cognee.

Inspired by DLT's schema contract system, Cognee contracts enforce rules at
multiple pipeline checkpoints: ingestion, graph extraction, and storage.
"""

from .models import (
    ContractMode,
    IngestionContract,
    GraphContract,
    DataContract,
)
from .violations import (
    ContractViolation,
    handle_violation,
    apply_graph_contract,
)
