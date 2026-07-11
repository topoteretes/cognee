from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EdgeIdentity:
    """Stable edge identifier used when provenance code addresses graph relationships."""

    source_id: str
    target_id: str
    relationship_name: str


@dataclass(frozen=True)
class NodeDeleteData:
    """
    Node payload needed to decide and perform graph-native delete or vector cleanup.

    Provenance fields have separate jobs:
    - source_ref_keys stores exact dataset/data ownership refs.
    - source_dataset_ids stores dataset ids for efficient dataset-delete filtering.
    - source_run_ids stores run ids for efficient rollback candidate filtering.
    - source_run_refs stores exact run/source-ref attachments for rollback.
    """

    node_id: str
    node_type: str
    indexed_fields: list[str]
    node_properties: dict[str, Any]
    source_ref_keys: list[str]
    source_dataset_ids: list[str]
    source_run_ids: list[str]
    source_run_refs: list[str]


@dataclass(frozen=True)
class EdgeDeleteData:
    """
    Edge payload needed to decide and perform graph-native delete or rollback.

    Provenance fields have separate jobs:
    - source_ref_keys stores exact dataset/data ownership refs.
    - source_dataset_ids stores dataset ids for efficient dataset-delete filtering.
    - source_run_ids stores run ids for efficient rollback candidate filtering.
    - source_run_refs stores exact run/source-ref attachments for rollback.
    """

    edge: EdgeIdentity
    edge_text: str
    edge_properties: dict[str, Any]
    source_ref_keys: list[str]
    source_dataset_ids: list[str]
    source_run_ids: list[str]
    source_run_refs: list[str]
