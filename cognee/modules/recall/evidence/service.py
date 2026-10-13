"""Resolve real graph artifacts; every expansion rechecks dataset read permission.

Never cache source bodies or return arbitrary node properties. With shared graph
storage (backend ACL disabled), dataset ownership cannot be proved for every
artifact, so this API explicitly reports unsupported instead of reading globally.
"""

import hashlib
import json
from collections import defaultdict
from urllib.parse import urlencode
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.engine import is_internal_node
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.models import Data
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

from .models import EvidenceIndex, EvidenceItem, EvidenceLocator

MAX_INDEX_ITEMS = 20
MAX_DETAIL_CHARS = 12000
logger = get_logger(__name__)


def _uuid(value):
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


def _node(value):
    # Adapters expose flat properties; tolerate the graph interface's nested form.
    props = value.get("properties")
    return {**value, **props} if isinstance(props, dict) else value


async def _documents(dataset_id, nodes):
    ids = {_uuid(n.get("document_id")) for n in nodes.values()} - {None}
    if not ids:
        return {}
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(Data).where(Data.id.in_(ids), Data.dataset_id == dataset_id)
                )
            )
            .scalars()
            .all()
        )
        return {
            str(row.id): {
                "data_id": str(row.id),
                "document_name": row.name,
                "source_version": row.content_hash,
                "source_ref_key": make_source_ref_key(dataset_id, row.id),
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "download_path": f"/api/v1/datasets/{dataset_id}/data/{row.id}/raw",
            }
            for row in rows
        }


def _material(ref, nodes, documents):
    if ref.kind == "graph_edge":
        left, right = nodes.get(str(ref.source_node_id)), nodes.get(str(ref.target_node_id))
        if not left or not right:
            return None
        title = f"{left.get('name', '')} --{ref.relationship_name}--> {right.get('name', '')}"
        return title, title, {"derived": True}
    node = nodes.get(str(ref.artifact_id))
    if not node:
        return None
    if ref.kind == "segment":
        doc = documents.get(str(node.get("document_id")))
        if not doc or not isinstance(node.get("text"), str):
            return None
        source = {**doc, "chunk_id": str(ref.artifact_id), "chunk_index": node.get("chunk_index")}
        return doc.get("document_name") or "Source passage", node["text"], source
    # Chunks must be resolved as segments so their Data membership is verified.
    if node.get("document_id") or node.get("type") == "DocumentChunk":
        return None
    title = str(node.get("name") or node.get("type") or "Graph entity")
    body = str(node.get("description") or node.get("text") or title)
    return title, body, {"derived": True}


def _item(ref, material, include_text):
    title, body, source = material
    # Data.updated_at can move when access/importance bookkeeping changes.
    # Only content/identity metadata participates in the citation's version.
    version_source = {k: v for k, v in source.items() if k not in {"updated_at", "created_at"}}
    version = hashlib.sha256(
        json.dumps([title, body, version_source], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    current = ref.model_copy(update={"version": version})
    changed = bool(ref.version and ref.version != version)
    citation = f"cognee://datasets/{ref.dataset_id}/evidence/{ref.kind}/{ref.artifact_id}@{version}"
    if ref.kind == "graph_edge":
        citation += "?" + urlencode(
            {
                "source_node_id": str(ref.source_node_id),
                "target_node_id": str(ref.target_node_id),
                "relationship_name": ref.relationship_name,
            }
        )
    return EvidenceItem(
        reference=current,
        status="changed" if changed else "available",
        current_version=version,
        requested_version=ref.version,
        citation=citation,
        title=title[:160],
        source=source,
        # A changed reference never silently supplies replacement text.
        text=body[:MAX_DETAIL_CHARS] if include_text and not changed else None,
        truncated=len(body) > MAX_DETAIL_CHARS,
    )


async def expand_evidence(
    references: list[EvidenceLocator], user: User, *, include_text: bool = True
) -> list[EvidenceItem]:
    """Internal batches also serve recall's bounded index; public batches cap at five."""
    if len(references) > MAX_INDEX_ITEMS:
        raise ValueError("Too many evidence references")
    if not references:
        return []
    result = [EvidenceItem(reference=ref, status="unavailable") for ref in references]
    # The explicit-ID helper is all-or-nothing. Read grants first so one revoked
    # dataset does not prevent the other requested artifacts from expanding.
    datasets = await get_authorized_existing_datasets(None, "read", user)
    permitted = {dataset.id: dataset for dataset in datasets}
    groups = defaultdict(list)
    for index, ref in enumerate(references):
        if ref.dataset_id in permitted:
            groups[ref.dataset_id].append((index, ref))
    for dataset_id, group in groups.items():
        try:
            if not backend_access_control_enabled():
                for index, _ in group:
                    result[index].status = "unsupported"
                continue
            dataset = permitted[dataset_id]
            async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                engine = await get_graph_engine()
                ids = set()
                for _, ref in group:
                    ids.update(
                        str(v)
                        for v in (ref.artifact_id, ref.source_node_id, ref.target_node_id)
                        if v
                    )
                nodes = {
                    str(node.get("id")): node
                    for raw in await engine.get_nodes(sorted(ids))
                    if not is_internal_node(node := _node(raw))
                }
                documents = await _documents(dataset_id, nodes)
                for index, ref in group:
                    if ref.kind == "graph_edge":
                        edge_id = generate_edge_object_id(
                            str(ref.source_node_id), str(ref.target_node_id), ref.relationship_name
                        )
                        if str(edge_id) != str(ref.artifact_id) or not await engine.has_edge(
                            str(ref.source_node_id), str(ref.target_node_id), ref.relationship_name
                        ):
                            continue
                    material = _material(ref, nodes, documents)
                    if material:
                        result[index] = _item(ref, material, include_text)
        except Exception:
            logger.exception("Evidence expansion failed for an authorized dataset")
            for index, ref in group:
                result[index] = EvidenceItem(reference=ref, status="error")
    return result


def collect_references(results):
    references, metadata, seen = [], [], set()
    unaddressable, omitted = 0, 0
    for result in results:
        row = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        raw_refs = row.get("metadata", {}).get("evidence", [])
        # Prefer source passages to derived entities in the compact index.
        raw_refs = sorted(raw_refs, key=lambda ref: ref.get("kind") != "segment")
        valid = 0
        for raw in raw_refs:
            try:
                ref = EvidenceLocator.model_validate(
                    {**raw, "dataset_id": raw.get("dataset_id") or row.get("dataset_id")}
                )
            except ValueError:
                continue
            valid += 1
            key = (ref.dataset_id, ref.kind, ref.artifact_id)
            if key in seen:
                continue
            seen.add(key)
            if len(references) == MAX_INDEX_ITEMS:
                omitted += 1
                continue
            references.append(ref)
            metadata.append({k: raw.get(k) for k in ("rank", "role", "assertion_id")})
        unaddressable += int(not valid)
    return references, metadata, omitted, unaddressable


async def evidence_index(results, user: User, *, include_text: bool = False) -> EvidenceIndex:
    refs, metadata, omitted, unaddressable = collect_references(results)
    items = await expand_evidence(refs, user, include_text=include_text)
    for item, meta in zip(items, metadata):
        for key, value in meta.items():
            setattr(item, key, value)
    return EvidenceIndex(items=items, omitted=omitted, unaddressable_results=unaddressable)


async def attach_evidence(results, user: User):
    """Attach exact, resolvable artifacts without pretending bundle lines are citations."""
    index = await evidence_index(results, user, include_text=True)
    for item in index.items:
        if item.text and len(item.text) > 1400:
            item.text = item.text[:1400]
            item.truncated = True
    by_key = {
        (i.reference.dataset_id, i.reference.kind, i.reference.artifact_id): i for i in index.items
    }
    for result in results:
        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        refs, _, _, _ = collect_references([result])
        metadata["evidence_items"] = [
            by_key[key].model_dump(mode="json")
            for ref in refs
            if (key := (ref.dataset_id, ref.kind, ref.artifact_id)) in by_key
        ]
        metadata["evidence_contract"] = "recall_evidence_v1"
    return results
