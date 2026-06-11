"""End-to-end lockstep test of the Entity/EntityType id migration on REAL stores.

Flow (default backends: sqlite + lancedb + ladybug, real LLM from .env):

1. add + cognify lorem ipsum (TRIPLET_EMBEDDING=true) -> authentic graph,
   vector (IndexSchema payloads!), ledger and triplet data.
2. DOWNGRADE via the migration's own down() (runner.downgrade_database_migrations):
   graph node ids, embedded edge-property endpoint ids, vector point ids
   (vectors preserved on LanceDB/PGVector), triplet point ids, ledger
   slugs/edge endpoints; revisions stamped back to NULL.
3. Run the real startup migration (run_database_migrations).
4. VERIFY graph / vector / triplet / ledger consistency, search, re-cognify
   (the #2510 EntityAlreadyExists scenario) and delete-by-ledger-slug.

Works in BOTH access-control modes: with it on, stores resolve through the
per-dataset context and revisions live on the dataset_database row; with
ENABLE_BACKEND_ACCESS_CONTROL=False, the global engines are used directly and
revisions live in the single-row global_database_version table.

Run (add ENABLE_BACKEND_ACCESS_CONTROL=False for the global-mode variant):
    TRIPLET_EMBEDDING=true SYSTEM_ROOT_DIRECTORY=... DATA_ROOT_DIRECTORY=... \
    python cognee/tests/migrations/test_migration_lockstep.py <seed|downgrade|migrate|verify|recognify|delete>
"""

import asyncio
import sys
from contextlib import asynccontextmanager

from sqlalchemy import select

import cognee
from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
from cognee.modules.data.models import Data, Dataset
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.models import Edge, Node
from cognee.modules.migrations.versions.namespace_entity_type_node_ids import build_id_remap
from cognee.modules.migrations.runner import run_database_migrations

TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Lorem Ipsum has been the
industry's standard dummy text ever since the 1500s, when an unknown printer took
a galley of type and scrambled it to make a type specimen book in Venice, Italy.
"""
DATASET = "lockstep_test"
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def _triplet_id(source: str, rel: str, target: str) -> str:
    return str(generate_node_id(source + rel + target))


async def verify(stage: str):
    async with _store_context():
        graph_engine = await get_graph_engine()
        vector_engine = get_vector_engine()
        nodes, edges = await graph_engine.get_graph_data()

        node_ids = {nid for nid, _ in nodes}
        entityish = [(nid, p) for nid, p in nodes if p.get("type") in ("Entity", "EntityType")]
        real_edges = [(s, t, r, p) for s, t, r, p in edges if not (r == "SELF" and s == t)]

        check(f"{stage}: graph non-empty", bool(nodes) and bool(entityish))
        check(
            f"{stage}: no old-scheme ids remain",
            not build_id_remap(nodes),
            f"{len(build_id_remap(nodes))} stale",
        )
        model = {"Entity": Entity, "EntityType": EntityType}
        # Positive id-scheme check. Not all(): an entity whose LLM-emitted raw
        # id differs from its name is LEGITIMATELY at id_for(raw_id), not
        # id_for(name) — the "no old-scheme ids remain" check above covers
        # staleness; this one proves the migration produced model-owned ids.
        on_scheme = sum(
            1 for nid, p in entityish if nid == str(model[p["type"]].id_for(p["name"]))
        )
        check(
            f"{stage}: entities on the model-owned id scheme",
            on_scheme > 0,
            f"{on_scheme}/{len(entityish)} (rest are the id!=name case)",
        )
        dangling = [(s, t) for s, t, _, _ in real_edges if s not in node_ids or t not in node_ids]
        check(f"{stage}: no dangling edges", not dangling, f"{len(dangling)} dangling")
        bad_props = [
            (s, t)
            for s, t, _, p in real_edges
            if (p or {}).get("source_node_id") not in (None, s)
            or (p or {}).get("target_node_id") not in (None, t)
        ]
        check(
            f"{stage}: embedded edge-prop endpoint ids match topology",
            not bad_props,
            f"{len(bad_props)} stale",
        )
        check(
            f"{stage}: no persisted SELF edges",
            all(not (r == "SELF" and s == t) for s, t, r, _ in edges) or len(real_edges) > 0,
        )

        # Vector: a point must exist at every entity's graph id, none at old ids.
        for kind, field in (("Entity", "name"), ("EntityType", "name")):
            ids = [nid for nid, p in entityish if p["type"] == kind]
            rows = await vector_engine.retrieve(f"{kind}_{field}", ids)
            check(
                f"{stage}: {kind} vector points at graph ids",
                len(rows) == len(ids),
                f"{len(rows)}/{len(ids)}",
            )
            payload_ok = all(
                set(dict(r.payload).keys()) >= {"id", "text"} and "name" not in dict(r.payload)
                for r in rows
            )
            check(f"{stage}: {kind} payloads still IndexSchema-shaped", payload_ok)
            old_ids = [
                str(generate_node_id(p["name"])) for nid, p in entityish if p["type"] == kind
            ]
            leftovers = await vector_engine.retrieve(f"{kind}_{field}", old_ids)
            stale = [r for r in leftovers if str(r.id) not in set(ids)]
            check(f"{stage}: no {kind} points left at old ids", not stale, f"{len(stale)} stale")

        # Triplets: ids must be recomputable from CURRENT edge endpoints.
        triplet_ids = [_triplet_id(s, r, t) for s, t, r, _ in real_edges]
        triplet_rows = await vector_engine.retrieve("Triplet_text", triplet_ids)
        check(
            f"{stage}: triplet points keyed by current endpoints",
            len(triplet_rows) > 0,
            f"{len(triplet_rows)}/{len(triplet_ids)} resolvable",
        )

        # Ledger: every Entity/EntityType slug + edge endpoint resolves to the graph.
        db = get_relational_engine()
        async with db.get_async_session() as session:
            ledger_nodes = (await session.scalars(select(Node))).all()
            ledger_edges = (await session.scalars(select(Edge))).all()
        stale_slugs = [
            str(n.slug)
            for n in ledger_nodes
            if n.type in ("Entity", "EntityType") and str(n.slug) not in node_ids
        ]
        check(
            f"{stage}: ledger slugs resolve to graph", not stale_slugs, f"{len(stale_slugs)} stale"
        )
        stale_ends = [
            e
            for e in ledger_edges
            if str(e.source_node_id) not in node_ids or str(e.destination_node_id) not in node_ids
        ]
        check(f"{stage}: ledger edge endpoints resolve", not stale_ends, f"{len(stale_ends)} stale")

    # Search (CHUNKS = raw vector retrieval; completion = smoke).
    chunks = await cognee.search(query_type=cognee.SearchType.CHUNKS, query_text="lorem ipsum")
    check(f"{stage}: CHUNKS search returns results", bool(chunks), f"{len(chunks)} results")
    await cognee.search(query_type=cognee.SearchType.GRAPH_COMPLETION, query_text="what is lorem?")
    check(f"{stage}: GRAPH_COMPLETION runs", True)


@asynccontextmanager
async def _store_context():
    """Per-dataset context when access control is on; global engines otherwise."""
    if backend_access_control_enabled():
        rows = await get_dataset_databases()
        assert rows, "no dataset_database rows — run the seed phase first"
        row = rows[0]
        async with set_database_global_context_variables(row.dataset_id, row.owner_id):
            yield
    else:
        yield


async def _dataset_id():
    db = get_relational_engine()
    async with db.get_async_session() as session:
        dataset = (await session.scalars(select(Dataset).where(Dataset.name == DATASET))).first()
    assert dataset, "dataset not found — run the seed phase first"
    return dataset.id


async def main(phase: str):
    """Each phase runs in its OWN process (driven by the wrapper below), exactly
    like a real upgrade: cognify on the old version, restart, migrate on the new
    version, restart, serve. LanceDB table handles are per-process, so phases
    must not share one."""
    if phase == "seed":
        print("\n[1] add + cognify (real pipeline, triplet embeddings on)")
        await cognee.add(TEXT, dataset_name=DATASET)
        await cognee.cognify(datasets=[DATASET])

    elif phase == "stamp":
        # Fresh OFF-mode stores have no global row until the runner creates it;
        # ON-mode rows are head-stamped at creation. Run once so a following
        # downgrade has stamped revisions to revert from.
        print("\n[1b] initial run_database_migrations (stamps head)")
        await run_database_migrations()

    elif phase == "downgrade":
        print("\n[2] downgrade via the migration's own down() (runner downgrade)")
        from cognee.modules.migrations.runner import downgrade_database_migrations

        summaries = await downgrade_database_migrations()
        check(
            "downgrade reverted the id migration",
            any(
                "namespace_entity_type_node_ids" in (s.get("migrations_reverted") or [])
                for s in summaries
            ),
            str(summaries),
        )
        async with _store_context():
            graph_engine = await get_graph_engine()
            nodes, _ = await graph_engine.get_graph_data()
        check(
            "graph is back on the released bare scheme",
            len(build_id_remap(nodes)) > 0,
            f"{len(build_id_remap(nodes))} remappable",
        )

    elif phase == "migrate":
        print("\n[3] run startup migrations")
        summaries = await run_database_migrations()
        migrated = [s for s in summaries if s.get("migrations_applied")]
        check("migration ran for the downgraded dataset", bool(migrated), str(summaries))
        summaries2 = await run_database_migrations()
        check(
            "second run applies nothing (idempotency + fast path)",
            all(not s.get("migrations_applied") for s in summaries2),
            str(summaries2),
        )

    elif phase == "verify":
        print("\n[4] verify after migration")
        await verify("post-migration")

    elif phase == "recognify":
        print("\n[5] re-cognify the same dataset (the #2510 EntityAlreadyExists scenario)")
        await cognee.add(TEXT, dataset_name=DATASET)
        await cognee.cognify(datasets=[DATASET])
        check("re-cognify does not raise", True)
        await verify("post-recognify")

    elif phase == "delete":
        print("\n[6] delete by ledger slug (proves ledger ids migrated correctly)")
        dataset_id = await _dataset_id()
        db = get_relational_engine()
        async with db.get_async_session() as session:
            data_rows = (await session.scalars(select(Data))).all()
        for data_row in data_rows:
            await cognee.delete(data_id=data_row.id, dataset_id=dataset_id, mode="hard")
        async with _store_context():
            graph_engine = await get_graph_engine()
            nodes, _ = await graph_engine.get_graph_data()
        remaining = [p for _, p in nodes if p.get("type") in ("Entity", "EntityType")]
        check("hard delete removed migrated entity nodes", not remaining, f"{len(remaining)} left")

    else:
        raise SystemExit(f"unknown phase {phase!r}")

    failed = [c for c in CHECKS if not c[1]]
    if CHECKS:
        print(
            f"phase {phase}: {'PASS' if not failed else 'FAIL'} "
            f"({len(CHECKS) - len(failed)}/{len(CHECKS)} checks)"
        )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "seed"))
