# Graph / Vector Data Migrations

Alembic-style revision chains for cognee's **graph and vector databases** (the
relational schema has its own Alembic setup under `cognee/alembic/`). Each
database records the slug of the last migration applied to it; on startup the
runner walks the chain forward.

## Where state lives

| Mode | Revisions | Written by |
|---|---|---|
| Access control ON (default) | `dataset_database.graph_migration_revision` / `.vector_migration_revision` — one row per dataset's database pair | head-stamped at row creation (`get_or_create_dataset_database`); advanced by `runner.run_database_migrations` |
| Access control OFF | `global_database_version.global_graph_migration_revision` / `.global_vector_migration_revision` — one row, one global pair | `runner._run_global_migrations` |

`cognee_version` columns are **audit-only** (which release last migrated /
last started); nothing gates on them. Revisions are the only gates.

## When migrations run

- FastAPI lifespan (`api/client.py`) — every server start, once per worker
- `remember()`'s first call in an SDK process
- explicitly: `await cognee.run_startup_migrations()`
- `cognify()` does NOT migrate, but warns when target databases have pending
  migrations

Steady state is cheap: an in-memory revision comparison per database, nothing
opened, nothing written.

## Operator commands (alembic-style)

```
cognee-cli upgrade [revision]              # head (default) or a slug; partial-chain ok
cognee-cli downgrade <revision> [--dataset ID]   # 'base' or a slug; data-rewriting, confirmed
cognee-cli stamp <revision> [--dataset ID] # set the stored revision WITHOUT running anything
cognee-cli history                         # the chains, newest first
cognee-cli current                         # each database's stamped revision
```

`stamp` is the bookkeeping-repair escape hatch: `stamp base --dataset <id>`
re-arms the chain for a database whose data drifted from its stamp (restored
backup, old-code writes after migration), letting the idempotent chain
converge it on the next upgrade.

## Authoring contract — read before writing migration #2

1. **Slugs are append-only and immutable.** The slug IS the revision stored in
   every deployed database. Renaming or removing a shipped entry orphans every
   stamped database: `pending_migrations` logs a warning and runs NOTHING for
   it, forever. Never "swap" an existing entry — append after it with
   `down_revision=<previous slug>`.
2. **Idempotent, and cheap on empty stores.** Fresh deployments execute the
   whole chain once (revisions start NULL); your migration must no-op quickly
   against an empty database, and re-running it against an already-migrated
   one must change nothing.
3. **Crash-retry safe, derived stores first.** If you rewrite ids that live in
   several stores, compute the map while the source of truth (the graph) still
   holds the source ids; re-key the derived stores (vector points, relational
   ledger) first and rename the graph LAST, so a crash at any point leaves a
   state the next startup finishes. See `graph/namespace_entity_type_node_ids.py`
   for the worked example.
4. **Freeze everything you derive.** Vendor private copies of id derivations,
   normalization rules and collection names into the migration module
   (`_frozen_*`). A revision must mean one deterministic transformation
   forever — never import live models (`Entity`, `id_for`, `index_fields`)
   into a migration; when the live scheme changes, that is a NEW migration.
5. **`MigrationContext`** carries `graph_engine`, `vector_engine`, and
   `dataset_id` (`None` in global mode → ledger updates unscoped). Get the
   relational engine yourself via `get_relational_engine()` (lazy import).
6. **Downgrades** are optional per migration (`down=`), explicit-only
   (`runner.downgrade_database_migrations`), and a chain can only be
   downgraded through a contiguous span where every entry defines `down`.
7. **Validate at import.** Each registry calls `order_migrations(...)` at
   module bottom so a typo'd `down_revision` fails in CI, not at customer
   startup. The unit tests additionally pin the shipped slugs.

## Landmine index (things migration #1 had to learn the hard way)

- Vector payloads are **`IndexSchema`-shaped** (`{id, text, belongs_to_set}`),
  written by `index_data_points` — never a dump of the source model. Re-insert
  through `index_data_points` with a small carrier; never reconstruct models
  from payloads.
- Re-keys must be **merge-safe**: the target id may already exist (an SDK
  process wrote new-scheme data before migrating). Delete the old row instead
  of moving it onto a duplicate key.
- Ladybug fabricates `(id, id, "SELF")` placeholder edges for edgeless graphs —
  filter them or you persist fake relationships.
- cognify embeds `source_node_id`/`target_node_id` INSIDE edge properties and
  retrieval prefers them over topology — remap them with the edge.
- `Triplet_text` point ids are hashed from edge endpoint ids — entity renames
  move triplet points too.
- The relational delete-ledger (`nodes.slug`, `edges.source/destination`) and
  the legacy `graph_relationship_ledger` key on graph node ids.
- Engines arrive wrapped (`_VectorEngineHandle`, leased proxies) — they spoof
  `__class__`, so dispatch on `engine.__class__.__name__`, never `type(engine)`.
- lance 0.32's `merge_insert` can panic on tables carrying deletion vectors —
  prefer plain `add` + delete in migrations, compact when the handle allows.
- Hybrid graph+vector backends (Neptune Analytics) store vectors as graph
  nodes sharing the entity id — id migrations must detect
  (`_is_hybrid_provider`) and refuse rather than corrupt.
- `get_graph_data()` loads the ENTIRE graph into memory; fine at current
  scales, budget for it in a migration that targets chunk-scale data.

## Concurrency

The runner serializes migrate-then-stamp per database with a Postgres
advisory lock held on a dedicated connection (no open transaction, so long
migrations don't block anything relational). SQLite has no cross-process
primitive: multi-worker startup against one SQLite metadata store during a
migration window is unsupported. Out-of-process writers (rolling deploys, SDK
scripts on old code) are NOT serialized against — ship id-scheme changes with
a maintenance window.

## Testing a migration

- Unit: drive the store-helpers against the fakes in
  `cognee/tests/unit/modules/migrations/` — they mirror the REAL adapter
  contracts (IndexSchema payloads; no `create_data_points` on the vector fake,
  so model-reconstruction regressions fail loudly).
- Integration: `scripts/test_migration_lockstep.py` — seed via real cognify,
  downgrade via the migration's own `down`, re-migrate, verify all stores,
  re-cognify, hard-delete. Run per phase in separate processes.
- Cross-version: `cognee/tests/backwards_compatibility/phase{1,2}` against a
  `main`-seeded database.
