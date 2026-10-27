"""Preserve references when legacy code symbols gain source-file identities."""

import json
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity

CODE_EDGE_OWNER = "cognee_code_graph"


class SymbolMigrationError(ValueError):
    """Unowned references cannot safely choose among a legacy symbol's successors."""

    def __init__(self, issues):
        self.issues = issues
        super().__init__(
            "Code symbol migration needs explicit reference resolution: " + json.dumps(issues)
        )


@dataclass
class SymbolMigration:
    edges: list
    replacements: dict[str, str]


def plan_symbol_migration(existing_nodes, existing_edges, data_points):
    from .extract_code_graph import fact_node_id

    candidates = defaultdict(list)
    for point in data_points:
        if getattr(point, "kind", None) == "symbol":
            candidates[(point.repo, point.name)].append(str(point.id))
    successors = {}
    for node_id, props in existing_nodes:
        if not isinstance(props, dict) or props.get("type") != "CodeSymbol":
            continue
        repo, name = props.get("repo"), props.get("name")
        if not isinstance(repo, str) or not isinstance(name, str):
            continue
        if str(node_id) != str(fact_node_id(repo, "symbol", name)):
            continue
        matches = sorted(set(candidates.get((repo, name), [])))
        if matches and str(node_id) not in matches:
            successors[str(node_id)] = matches

    node_types = {
        str(key): props.get("type") for key, props in existing_nodes if isinstance(props, dict)
    }
    transfers, issues, replacements = [], [], {}
    for source, target, relation, properties in existing_edges:
        source, target = str(source), str(target)
        properties = properties if isinstance(properties, dict) else {}
        if (
            properties.get("producer") == CODE_EDGE_OWNER
            or (relation == "part_of" and node_types.get(target) == "CodeRepository")
            or (relation == "belongs_to_set" and node_types.get(target) == "NodeSet")
        ):
            continue
        endpoints = [node for node in (source, target) if node in successors]
        if not endpoints:
            continue
        ambiguous = [node for node in endpoints if len(successors[node]) != 1]
        if ambiguous:
            issues.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "relationship": relation,
                    "candidates": {node: successors[node] for node in ambiguous},
                }
            )
            continue
        for node in endpoints:
            replacements[node] = successors[node][0]
        new_source, new_target = replacements.get(source, source), replacements.get(target, target)
        props = dict(properties)
        for field, value in (("source_node_id", new_source), ("target_node_id", new_target)):
            if field in props:
                props[field] = value
        transfers.append((new_source, new_target, relation, props))
    if issues:
        raise SymbolMigrationError(issues)
    return SymbolMigration(transfers, replacements)


async def transfer_symbol_references(graph, migration, existing_edges, ctx=None):
    """Copy edges and their original ownership before any old node is removed."""
    if not migration.edges:
        return
    from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
    from cognee.infrastructure.databases.provenance.source_refs import (
        get_pipeline_run_id_from_source_run_ref,
        get_source_ref_key_from_source_run_ref,
    )

    pairs = []
    new_keys = {(str(s), str(t), r) for s, t, r, _ in migration.edges}
    for source, target, relation, _ in existing_edges:
        source, target = str(source), str(target)
        new_source = migration.replacements.get(source, source)
        new_target = migration.replacements.get(target, target)
        if (source, target) != (new_source, new_target) and (
            new_source,
            new_target,
            relation,
        ) in new_keys:
            pairs.append(
                (
                    EdgeIdentity(source, target, relation),
                    EdgeIdentity(new_source, new_target, relation),
                )
            )
    graph_provenance = await stores_provenance_in_graph(graph)
    ownership = (
        await graph.get_edge_delete_data([old for old, _ in pairs]) if graph_provenance else {}
    )
    await graph.add_edges(migration.edges)
    if graph_provenance:
        for old, new in pairs:
            stored = ownership.get(old)
            if stored is None:
                continue
            attached = set()
            for run_ref in stored.source_run_refs:
                ref = get_source_ref_key_from_source_run_ref(run_ref)
                await graph.attach_edge_source_refs(
                    [new],
                    [ref],
                    pipeline_run_id=str(get_pipeline_run_id_from_source_run_ref(run_ref)),
                )
                attached.add(ref)
            remaining = sorted(set(stored.source_ref_keys) - attached)
            if remaining:
                await graph.attach_edge_source_refs([new], remaining)
    elif ctx is not None and getattr(ctx, "dataset", None) is not None:
        await _transfer_ledger(pairs, ctx.dataset.id)


async def _transfer_ledger(pairs, dataset_id):
    """Keep source data/user/run ownership instead of assigning the repository's run."""
    from sqlalchemy import and_, delete, or_, select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.graph.methods.upsert_edges import upsert_edges
    from cognee.modules.graph.models.Edge import Edge
    from cognee.modules.users.models import User

    mapping = {(old.source_id, old.target_id, old.relationship_name): new for old, new in pairs}
    conditions = [
        and_(Edge.source_node_id == UUID(s), Edge.destination_node_id == UUID(t), Edge.label == r)
        for s, t, r in mapping
    ]
    if not conditions:
        return
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(Edge).where(Edge.dataset_id == dataset_id, or_(*conditions))
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            new = mapping[(str(row.source_node_id), str(row.destination_node_id), row.label)]
            user = await session.get(User, row.user_id)
            if user is None:
                raise ValueError("Cannot migrate an edge whose original owner is missing.")
            attributes = dict(row.attributes or {})
            for key, value in (
                ("source_node_id", new.source_id),
                ("target_node_id", new.target_id),
            ):
                if key in attributes:
                    attributes[key] = value
            await upsert_edges(
                [(UUID(new.source_id), UUID(new.target_id), new.relationship_name, attributes)],
                tenant_id=user.tenant_id,
                user_id=row.user_id,
                data_id=row.data_id,
                dataset_id=row.dataset_id,
                pipeline_run_id=row.pipeline_run_id,
                session=session,
            )
            await session.execute(
                delete(Edge).where(Edge.id == row.id, Edge.dataset_id == dataset_id)
            )
        await session.commit()
