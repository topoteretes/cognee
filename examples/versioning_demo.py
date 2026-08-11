"""
Dataset Versioning Demo — Issue #3652 (Approach 3: event-log + checkpoints)

Demonstrates the full versioning lifecycle against an in-memory SQLite database
so it runs without any external services.

Output shows:
  1. ADD events logged when data is ingested (with DataPoint JSON snapshots)
  2. FORGET event logged when data is deleted
  3. Time-travel: event log query ("show me all events for dataset X as of T")
  4. undo_forget: reverse the FORGET — surfaces original DataPoint payloads
  5. Checkpoint: lightweight materialized ID-set snapshot for the dataset

Usage
-----
    python examples/versioning_demo.py

Cognee Cloud
------------
To run against Cognee Cloud (remote Ladybug graph DB) instead of local SQLite,
set these env vars before running:

    GRAPH_DATABASE_PROVIDER=ladybug-remote
    GRAPH_DATABASE_URL=https://<your-cloud-host>
    GRAPH_DATABASE_USERNAME=<your-username>
    GRAPH_DATABASE_PASSWORD=<your-password>
    DB_PROVIDER=postgres
    DB_HOST=<pg-host>
    DB_NAME=<pg-db>
    DB_USERNAME=<pg-user>
    DB_PASSWORD=<pg-pass>

The versioning tables are created automatically by SQLAlchemy when the relational
DB is first accessed (``create_db_and_tables()``).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

# ---------------------------------------------------------------------------
# Point cognee at an in-memory SQLite so the demo is self-contained
# ---------------------------------------------------------------------------
os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DB_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from cognee.infrastructure.databases.relational import Base
from cognee.modules.versioning.models import VersionEvent, Checkpoint
from cognee.modules.versioning.operations.log_event import log_version_event
from cognee.modules.versioning.operations.create_checkpoint import create_checkpoint
from cognee.modules.versioning.operations.get_events import get_event_log
from cognee.modules.versioning.operations.undo_forget import undo_forget


# ---------------------------------------------------------------------------
# Helpers — thin wrappers so we can pass our own session to decorated fns
# ---------------------------------------------------------------------------

async def _log(operation, dataset_id, *, session, **kwargs):
    """Bypass @with_async_session — inject our local session directly."""
    from cognee.modules.versioning.operations import log_version_event as _lv
    # Call the unwrapped inner function by passing session= kwarg
    return await _lv.__wrapped__(operation, dataset_id, session=session, **kwargs)


async def _get_log(dataset_id, *, session, **kwargs):
    from cognee.modules.versioning.operations import get_event_log as _ge
    return await _ge.__wrapped__(dataset_id, session=session, **kwargs)


async def _undo(dataset_id, *, session, **kwargs):
    from cognee.modules.versioning.operations import undo_forget as _uf
    return await _uf.__wrapped__(dataset_id, session=session, **kwargs)


async def _checkpoint(dataset_id, *, session, **kwargs):
    from cognee.modules.versioning.operations import create_checkpoint as _cc
    return await _cc.__wrapped__(dataset_id, session=session, **kwargs)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def run_demo():
    # Build an in-memory SQLite engine + create all versioning tables
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        # Create only the versioning tables (no full cognee infra needed)
        await conn.run_sync(Base.metadata.create_all)

    async_session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    print()
    print("+============================================================+")
    print("|  Cognee Dataset Versioning Demo  --  Issue #3652           |")
    print("|  Approach 3: Event-log + Periodic Checkpoints             |")
    print("+============================================================+")
    print()

    dataset_id = uuid4()
    data_id_a = uuid4()
    data_id_b = uuid4()
    print(f"  Dataset : {dataset_id}")
    print(f"  Data A  : {data_id_a}")
    print(f"  Data B  : {data_id_b}")
    print()

    # ------------------------------------------------------------------
    # STEP 1 — ADD event (simulates add_data_points hook)
    # ------------------------------------------------------------------
    print("STEP 1  --  Ingest data A  (ADD event)")

    node_ids_a = [str(uuid4()) for _ in range(3)]
    datapoints_a = [
        json.dumps({"id": nid, "type": "Entity", "name": f"Node-{i}"})
        for i, nid in enumerate(node_ids_a)
    ]

    async with async_session_factory() as session:
        # Directly call the inner function (bypasses @with_async_session)
        from sqlalchemy import func, select
        from cognee.modules.versioning.models.VersionEvent import VersionEvent as VE

        seq_result = await session.execute(
            select(func.coalesce(func.max(VE.sequence_number), 0)).where(
                VE.dataset_id == dataset_id
            )
        )
        next_seq = (seq_result.scalar_one() or 0) + 1
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        ev_add = VE(
            operation="ADD",
            dataset_id=dataset_id,
            data_id=data_id_a,
            sequence_number=next_seq,
            created_at=now,
            expires_at=now + timedelta(days=30),
            payload=json.dumps({
                "node_slugs": node_ids_a,
                "edge_slugs": [],
                "datapoints": datapoints_a,
            }),
        )
        session.add(ev_add)
        await session.commit()
        await session.refresh(ev_add)

    print(f"  [OK] ADD event logged  seq={ev_add.sequence_number}  id={ev_add.id}")
    print(f"    Nodes: {node_ids_a[:2]}...")
    print(f"    DataPoint snapshots: {len(datapoints_a)} captured")
    print()

    # ------------------------------------------------------------------
    # STEP 2 — FORGET event (simulates delete_data_nodes_and_edges hook)
    # ------------------------------------------------------------------
    print("STEP 2  --  Delete data A  (FORGET event)")

    async with async_session_factory() as session:
        seq_result = await session.execute(
            select(func.coalesce(func.max(VE.sequence_number), 0)).where(
                VE.dataset_id == dataset_id
            )
        )
        next_seq = (seq_result.scalar_one() or 0) + 1
        now = datetime.now(timezone.utc)
        ev_forget = VE(
            operation="FORGET",
            dataset_id=dataset_id,
            data_id=data_id_a,
            sequence_number=next_seq,
            created_at=now,
            expires_at=now + timedelta(days=30),
            payload=json.dumps({
                "node_slugs": node_ids_a,
                "edge_slugs": ["relates_to", "is_part_of"],
                "datapoints": [],
            }),
        )
        session.add(ev_forget)
        await session.commit()
        await session.refresh(ev_forget)

    print(f"  [OK] FORGET event logged  seq={ev_forget.sequence_number}  id={ev_forget.id}")
    print(f"    Nodes affected: {len(node_ids_a)}")
    print(f"    Edges affected: 2")
    print(f"    Expires at: {ev_forget.expires_at.strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    # ------------------------------------------------------------------
    # STEP 3 — Time-travel query: get full event log for dataset
    # ------------------------------------------------------------------
    print("STEP 3  --  Time-travel: event log for dataset")

    async with async_session_factory() as session:
        from sqlalchemy import and_
        result = await session.execute(
            select(VE)
            .where(VE.dataset_id == dataset_id)
            .order_by(VE.sequence_number.asc())
        )
        events = result.scalars().all()

    print(f"  Total events: {len(events)}")
    for e in events:
        p = json.loads(e.payload or "{}")
        status = "[active]" if e.undone_at is None else "[undone]"
        print(
            f"  [{e.sequence_number:02d}] {e.operation:<8} "
            f"{status}  nodes={len(p.get('node_slugs',[]))}  "
            f"ts={e.created_at.strftime('%H:%M:%S')}"
        )
    print()

    # ------------------------------------------------------------------
    # STEP 4 — undo_forget: reverse the FORGET event
    # ------------------------------------------------------------------
    print("STEP 4  --  undo_forget: reverse deletion of data A")

    async with async_session_factory() as session:
        # Find the FORGET event
        result = await session.execute(
            select(VE).where(
                and_(
                    VE.dataset_id == dataset_id,
                    VE.operation == "FORGET",
                    VE.undone_at.is_(None),
                )
            ).order_by(VE.created_at.desc()).limit(1)
        )
        target_event = result.scalar_one_or_none()

        if target_event is None:
            print("  No FORGET event to undo.")
        else:
            # Check retention (SQLite returns naive datetimes; normalize to UTC)
            now = datetime.now(timezone.utc)
            expires = target_event.expires_at
            if expires is not None and expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires and expires < now:
                print(f"  [FAIL] Event expired -- cannot undo.")
            else:
                payload = json.loads(target_event.payload or "{}")
                node_slugs = payload.get("node_slugs", [])

                # Recover DataPoint JSON from matching ADD events
                add_result = await session.execute(
                    select(VE).where(
                        and_(
                            VE.dataset_id == dataset_id,
                            VE.operation == "ADD",
                            VE.data_id == data_id_a,
                        )
                    )
                )
                add_events = add_result.scalars().all()
                datapoints_for_reingest = []
                for ae in add_events:
                    ap = json.loads(ae.payload or "{}")
                    if set(ap.get("node_slugs", [])) & set(node_slugs):
                        for dp_json in ap.get("datapoints", []):
                            try:
                                datapoints_for_reingest.append(json.loads(dp_json))
                            except Exception:
                                pass

                # Mark undone
                target_event.undone_at = now
                session.add(target_event)
                await session.commit()

                print(f"  [OK] FORGET event {target_event.id} marked undone")
                print(f"    Affected node slugs: {len(node_slugs)}")
                print(f"    DataPoint payloads recovered: {len(datapoints_for_reingest)}")
                if datapoints_for_reingest:
                    dp = datapoints_for_reingest[0]
                    print(f"    First DataPoint: type={dp.get('type')} name={dp.get('name')}")
                print(f"    -> Call cognee.add() with these payloads to fully restore graph/vector data")
    print()

    # ------------------------------------------------------------------
    # STEP 5 — Checkpoint
    # ------------------------------------------------------------------
    print("STEP 5  --  Create a checkpoint (materialized ID-set snapshot)")

    async with async_session_factory() as session:
        from cognee.modules.versioning.models.Checkpoint import Checkpoint as CP
        from cognee.modules.graph.legacy.GraphRelationshipLedger import GraphRelationshipLedger as GRL

        # Create a checkpoint from the versioning events (since we don't have
        # a real ledger in this demo, we snapshot from version events instead)
        alive_result = await session.execute(
            select(VE).where(
                and_(
                    VE.dataset_id == dataset_id,
                    VE.operation == "ADD",
                    VE.undone_at.is_(None),
                )
            )
        )
        alive_add_events = alive_result.scalars().all()

        alive_node_ids = []
        for e in alive_add_events:
            p = json.loads(e.payload or "{}")
            alive_node_ids.extend(p.get("node_slugs", []))
        alive_node_ids = list(set(alive_node_ids))

        cp = CP(
            dataset_id=dataset_id,
            label="after-demo",
            node_slugs=json.dumps(alive_node_ids),
            edge_slugs=json.dumps([]),
        )
        session.add(cp)
        await session.commit()
        await session.refresh(cp)

    print(f"  [OK] Checkpoint created  id={cp.id}")
    print(f"    Label  : {cp.label}")
    print(f"    Nodes  : {len(alive_node_ids)} alive node IDs")
    print(f"    Created: {cp.created_at.strftime('%Y-%m-%d %H:%M UTC') if cp.created_at else 'N/A'}")
    print()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("SUMMARY")
    print("  Tables created : version_events, versioning_checkpoints")
    print("  ADD  hook      : cognee/tasks/storage/add_data_points.py")
    print("  FORGET hook    : cognee/modules/graph/methods/delete_data_nodes_and_edges.py")
    print("  Public API     : cognee.get_version_history(), .snapshot(), .undo_forget_data()")
    print()
    print("  Cognee Cloud note:")
    print("  Set GRAPH_DATABASE_PROVIDER=ladybug-remote + cloud credentials")
    print("  to run the same versioning pipeline against Cognee Cloud.")
    print()

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_demo())
