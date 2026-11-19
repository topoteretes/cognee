"""Replace one document's node-set membership in graph and vector storage.

A node set is an organizational filter, stored per artifact in three places:
``belongs_to_set`` edges to NodeSet nodes (graph-side filtering), the
``belongs_to_set`` name array on the node and on its vector payload (what chunk
searches filter on), and the ``source_node_set`` / ``external_metadata["node_set"]``
copies on the document and its chunks. Reassignment rewrites only those, from
the document's in-graph ownership: nothing is extracted, chunked, embedded, or
deleted and recreated, and content nodes and non-membership edges keep their
identities and provenance.

Which nodes carry the document's membership, and under which owner refs:

- the source of every ``belongs_to_set`` edge the document owns, under the refs
  it owns on that edge (its existing contribution);
- the document node itself, under the document-scoped ref;
- its chunks (``is_part_of`` the document), the summaries ``made_from`` them and
  the entities they ``contain`` — the nodes ingestion tags from their chunk —
  under the chunk refs on those edges.

The second list is what makes membership recoverable after a clear (which
leaves no membership edges) and for a document ingested without a node set.

Shared artifacts keep every other contributor's membership: the document's
refs are detached from a membership edge, which is deleted only when no other
ref remains; a name leaves a node's tag array only when the node's membership
edge to that set is gone; tags the document never contributed are kept. A
NodeSet node is deleted only when it loses its last owner, as deletion would.

``EdgeType`` vectors are keyed by edge text shared across edges and carry no
membership, so they are neither written for new membership edges (that would
embed) nor removed for retired ones.
"""

import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.infrastructure.databases.unified.document_source_refs import (
    find_document_source_refs,
)
from cognee.modules.chunking.external_metadata import (
    decode_external_metadata,
    normalize_external_metadata,
)
from cognee.modules.data.methods.publish_updated_data import with_node_set
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.exceptions import NodeSetReassignmentNotSupportedError
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties

MEMBERSHIP = "belongs_to_set"

RefKeys = tuple[str, ...]


@dataclass
class NodeSetReassignment:
    """Every write that moves one document to its new node sets, in apply order.

    Empty when storage already matches the requested assignment. Groupings keyed
    by ``RefKeys`` collect artifacts that take the same owner refs, so each
    group is one adapter call.
    """

    document_ref: str
    nodesets_to_create: list[NodeSet] = field(default_factory=list)
    nodesets_to_own: list[str] = field(default_factory=list)
    edges_to_create: dict[RefKeys, list[tuple[str, str, str, dict]]] = field(default_factory=dict)
    edge_refs_to_attach: dict[RefKeys, list[EdgeIdentity]] = field(default_factory=dict)
    node_patches: dict[str, dict[str, Any]] = field(default_factory=dict)
    payload_patches: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    edge_refs_to_remove: dict[RefKeys, list[EdgeIdentity]] = field(default_factory=dict)
    edges_to_delete: list[EdgeIdentity] = field(default_factory=list)
    nodeset_refs_to_remove: dict[RefKeys, list[str]] = field(default_factory=dict)
    nodesets_to_delete: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(
            getattr(self, item.name) for item in fields(self) if item.name != "document_ref"
        )


def _set_id(name: str) -> str | None:
    """The NodeSet id a tag names, or None for a tag no NodeSet can have."""
    try:
        return str(NodeSet.id_for(name))
    except UnicodeEncodeError:
        return None


def _retagged(tags: Any, removed_ids: set[str], node_set: list[str]) -> list:
    """``tags`` without the removed sets, followed by the new sets it lacks."""
    current = list(tags) if isinstance(tags, list) else []
    kept = [tag for tag in current if not (isinstance(tag, str) and _set_id(tag) in removed_ids)]
    kept_ids = {_set_id(tag) for tag in kept if isinstance(tag, str)}
    return kept + [name for name in node_set if _set_id(name) not in kept_ids]


def _metadata_text(stored: Any, node_set: list[str], is_document: bool) -> str | None:
    """A stored ``external_metadata`` text with ``node_set`` replaced.

    The document stores its metadata as ``classify_documents`` writes it (indented
    JSON); a chunk stores the normalized copy chunks carry (None when empty).
    """
    metadata = with_node_set(decode_external_metadata(stored), node_set)
    if is_document:
        return json.dumps(metadata, indent=4)
    return normalize_external_metadata(metadata)


def _node_patch(
    properties: dict[str, Any],
    removed_ids: set[str],
    node_set: list[str],
    is_document: bool,
    is_chunk: bool,
) -> dict[str, Any]:
    """The membership properties of one graph node that must change, and their values."""
    patch: dict[str, Any] = {}
    stored_tags = properties.get(MEMBERSHIP)
    tags = _retagged(stored_tags, removed_ids, node_set)
    if tags != (list(stored_tags) if isinstance(stored_tags, list) else []):
        patch[MEMBERSHIP] = tags
    # classify_documents always records the document's sets here; other nodes
    # carry it only when something wrote it, and only then is it kept in step.
    if is_document or properties.get("source_node_set"):
        joined = ", ".join(tag for tag in tags if isinstance(tag, str)) or None
        if properties.get("source_node_set") != joined:
            patch["source_node_set"] = joined
    if is_document or is_chunk:
        stored = properties.get("external_metadata")
        current = decode_external_metadata(stored) or {}
        if current != with_node_set(current, node_set):
            patch["external_metadata"] = _metadata_text(stored, node_set, is_document)
    return patch


def _membership_edge_properties(edge: EdgeIdentity) -> dict[str, Any]:
    """The properties ``get_graph_from_model`` gives a ``belongs_to_set`` edge."""
    return {
        "source_node_id": edge.source_id,
        "target_node_id": edge.target_id,
        "relationship_name": edge.relationship_name,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def _group(groups: dict, key, item) -> None:
    groups.setdefault(key, []).append(item)


def _membership_carriers(
    document_id: str,
    document_ref: str,
    refs_by_node: dict[str, list[str]],
    refs_by_edge: dict[EdgeIdentity, list[str]],
    nodes: dict,
) -> tuple[dict[str, list[str]], set[str]]:
    """``({node id: owner refs of its membership}, chunk ids)``, see the module docstring."""
    carriers: dict[str, list[str]] = {}

    def carry(node_id: str, refs: list[str]) -> None:
        if node_id not in nodes:
            return
        owners = carriers.setdefault(node_id, [])
        owners.extend(ref for ref in refs if ref not in owners)

    def node_type(node_id: str) -> str | None:
        node = nodes.get(node_id)
        return node.node_type if node else None

    def carrier_of(edge: EdgeIdentity) -> str | None:
        """The node this owned edge gives membership to, if any."""
        relationship = edge.relationship_name
        if relationship == MEMBERSHIP and node_type(edge.target_id) == NodeSet.__name__:
            return edge.source_id
        if relationship == "is_part_of" and edge.target_id == document_id:
            return edge.source_id
        if relationship == "made_from" and edge.target_id in chunk_ids:
            return edge.source_id
        if (
            relationship == "contains"
            and edge.source_id in chunk_ids
            and node_type(edge.target_id) == Entity.__name__
        ):
            return edge.target_id
        return None

    if document_id in refs_by_node:
        carry(document_id, [document_ref])
    chunk_ids = {
        edge.source_id
        for edge in refs_by_edge
        if edge.relationship_name == "is_part_of" and edge.target_id == document_id
    }
    for edge, refs in refs_by_edge.items():
        if (carrier := carrier_of(edge)) is not None:
            carry(carrier, refs)
    return carriers, chunk_ids


async def plan_node_set_reassignment(
    graph,
    vector,
    dataset_id,
    data_id,
    node_set: list[str],
    stored_node_set: list[str],
) -> NodeSetReassignment:
    """Read the document's ownership and decide every write; write nothing.

    ``node_set`` is the complete replacement (validated, deduplicated);
    ``stored_node_set`` is the relational record of the current assignment,
    which keeps a retry after a partial write aimed at the sets it is leaving.

    Raises:
        NodeSetReassignmentNotSupportedError: an indexed vector row has no
            ``belongs_to_set`` payload field to patch.
    """
    document_id = str(data_id)
    document_ref = make_source_ref_key(dataset_id, data_id)
    plan = NodeSetReassignment(document_ref=document_ref)

    refs_by_node, refs_by_edge = await find_document_source_refs(graph, dataset_id, data_id)
    if not refs_by_node and not refs_by_edge:
        return plan  # Never cognified: no stored memory to reconcile.

    new_ids = {_set_id(name): name for name in node_set}
    stored_ids = {_set_id(name) for name in stored_node_set} - {None}
    nodes = await graph.get_node_delete_data(
        list(
            set(refs_by_node)
            | {edge.source_id for edge in refs_by_edge}
            | {edge.target_id for edge in refs_by_edge}
            | set(new_ids)
            | stored_ids
        )
    )
    carriers, chunk_ids = _membership_carriers(
        document_id, document_ref, refs_by_node, refs_by_edge, nodes
    )

    def is_nodeset(node_id: str) -> bool:
        return node_id in nodes and nodes[node_id].node_type == NodeSet.__name__

    owned_nodesets = {node_id for node_id in refs_by_node if is_nodeset(node_id)}
    member_of = {
        edge.target_id
        for edge in refs_by_edge
        if edge.relationship_name == MEMBERSHIP and is_nodeset(edge.target_id)
    }
    stale_ids = (owned_nodesets | member_of | stored_ids) - set(new_ids)

    # NodeSet nodes: the document owns each set it belongs to (document-scoped),
    # as ingestion stamps them, and stops owning the ones it leaves.
    if carriers:
        for set_id, name in new_ids.items():
            if set_id not in nodes:
                plan.nodesets_to_create.append(NodeSet(name=name))
            elif document_ref not in nodes[set_id].source_ref_keys:
                plan.nodesets_to_own.append(set_id)
    for set_id in sorted(stale_ids & owned_nodesets):
        owned_refs = refs_by_node[set_id]
        if set(nodes[set_id].source_ref_keys) <= set(owned_refs):
            plan.nodesets_to_delete.append(set_id)
        else:
            _group(plan.nodeset_refs_to_remove, tuple(owned_refs), set_id)

    existing_set_ids = [set_id for set_id in list(new_ids) + sorted(stale_ids) if set_id in nodes]
    existing_edges = await graph.get_edge_delete_data(
        [
            EdgeIdentity(node_id, set_id, MEMBERSHIP)
            for node_id in carriers
            for set_id in existing_set_ids
        ]
    )

    new_edges: list[tuple[str, str, str, dict]] = []
    new_edge_owners: list[RefKeys] = []
    removed_by_node: dict[str, set[str]] = {}
    for node_id, owner_refs in carriers.items():
        removed_ids = set()
        for set_id in stale_ids:
            edge = EdgeIdentity(node_id, set_id, MEMBERSHIP)
            stored_edge = existing_edges.get(edge)
            owned_refs = refs_by_edge.get(edge, [])
            if stored_edge is None:
                removed_ids.add(set_id)
            elif owned_refs and set(stored_edge.source_ref_keys) <= set(owned_refs):
                plan.edges_to_delete.append(edge)
                removed_ids.add(set_id)
            elif owned_refs:
                _group(plan.edge_refs_to_remove, tuple(owned_refs), edge)
        for set_id in new_ids:
            edge = EdgeIdentity(node_id, set_id, MEMBERSHIP)
            stored_edge = existing_edges.get(edge)
            if stored_edge is None:
                new_edges.append((node_id, set_id, MEMBERSHIP, _membership_edge_properties(edge)))
                new_edge_owners.append(tuple(owner_refs))
            elif missing := [ref for ref in owner_refs if ref not in stored_edge.source_ref_keys]:
                _group(plan.edge_refs_to_attach, tuple(missing), edge)

        removed_by_node[node_id] = removed_ids
        patch = _node_patch(
            nodes[node_id].node_properties,
            removed_ids,
            node_set,
            is_document=node_id == document_id,
            is_chunk=node_id in chunk_ids,
        )
        if patch:
            plan.node_patches[node_id] = patch

    # Edge texts are derived the way ingestion derives them, from both endpoints.
    # Metadata comes from the snapshot's parsed index fields: adapters that store
    # it as JSON text would hand the label builder a string.
    labels = [
        {
            **nodes[node_id].node_properties,
            "metadata": {"index_fields": nodes[node_id].indexed_fields},
        }
        for node_id in carriers
    ]
    labels += [{"id": set_id, "name": name} for set_id, name in new_ids.items()]
    for owner_refs, new_edge in zip(
        new_edge_owners, ensure_default_edge_properties(new_edges, nodes=labels)
    ):
        _group(plan.edges_to_create, owner_refs, new_edge)

    await _plan_payload_patches(plan, vector, nodes, removed_by_node, chunk_ids, node_set)
    return plan


async def _plan_payload_patches(
    plan: NodeSetReassignment,
    vector,
    nodes: dict,
    removed_by_node: dict[str, set[str]],
    chunk_ids: set[str],
    node_set: list[str],
) -> None:
    """Patch the vector rows of every carrier, from each row's own stored payload.

    Vector rows union ``belongs_to_set`` across writers while graph nodes keep
    the last writer's, so each representation is retagged from its own value.
    """
    ids_by_collection: dict[str, list[str]] = {}
    for node_id in removed_by_node:
        node = nodes[node_id]
        for index_field in node.indexed_fields:
            _group(ids_by_collection, f"{node.node_type}_{index_field}", node_id)

    for collection, node_ids in ids_by_collection.items():
        if not await vector.has_collection(collection):
            continue
        for row in await vector.retrieve(collection, node_ids):
            node_id = str(row.id)
            payload = row.payload or {}
            if node_id not in removed_by_node:
                continue
            if MEMBERSHIP not in payload:
                raise NodeSetReassignmentNotSupportedError(
                    f"vector collection {collection!r} stores no {MEMBERSHIP!r} payload "
                    "field, so its rows cannot be reassigned without re-embedding"
                )
            patch: dict[str, Any] = {}
            stored_tags = payload[MEMBERSHIP]
            tags = _retagged(stored_tags, removed_by_node[node_id], node_set)
            if tags != (list(stored_tags) if isinstance(stored_tags, list) else []):
                patch[MEMBERSHIP] = tags
            if node_id in chunk_ids and "external_metadata" in payload:
                stored = payload["external_metadata"]
                current = decode_external_metadata(stored) or {}
                if current != with_node_set(current, node_set):
                    patch["external_metadata"] = _metadata_text(stored, node_set, is_document=False)
            if patch:
                plan.payload_patches.setdefault(collection, {})[node_id] = patch


async def apply_node_set_reassignment(graph, vector, plan: NodeSetReassignment) -> None:
    """Execute ``plan``: additions first, then patches, then retirements.

    Every step is idempotent and the plan is re-derived from stored ownership
    on each call, so a call that fails part-way is completed by repeating it:
    the replacement memberships it already added count as the document's, and
    the ones it still has to retire are found again.
    """
    if plan.nodesets_to_create:
        await graph.add_nodes(plan.nodesets_to_create, source_ref_key=plan.document_ref)
    if plan.nodesets_to_own:
        await graph.attach_node_source_refs(plan.nodesets_to_own, [plan.document_ref])

    for owner_refs, edges in plan.edges_to_create.items():
        await graph.add_edges(edges, source_ref_key=owner_refs[0])
        if len(owner_refs) > 1:
            await graph.attach_edge_source_refs(
                [EdgeIdentity(*edge[:3]) for edge in edges], list(owner_refs[1:])
            )
    for owner_refs, edges in plan.edge_refs_to_attach.items():
        await graph.attach_edge_source_refs(edges, list(owner_refs))

    for node_id, patch in plan.node_patches.items():
        if not await graph.update_node(node_id, patch):
            raise RuntimeError(f"node {node_id} disappeared while its node sets were reassigned")
    for collection, patches in plan.payload_patches.items():
        await vector.update_payload(collection, patches)

    for owner_refs, edges in plan.edge_refs_to_remove.items():
        await graph.remove_edge_source_refs(edges, list(owner_refs))
    if plan.edges_to_delete:
        await graph.delete_edge_triples(plan.edges_to_delete)
    for owner_refs, node_ids in plan.nodeset_refs_to_remove.items():
        await graph.remove_node_source_refs(node_ids, list(owner_refs))
    if plan.nodesets_to_delete:
        await graph.delete_nodes(plan.nodesets_to_delete)
