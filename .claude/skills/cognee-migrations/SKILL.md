---
name: cognee-migrations
description: Use when dealing with cognee database migrations — understanding when they run automatically, checking or repairing migration state with cognee-cli upgrade/downgrade/stamp/current, a write blocked by a failed migration, authoring a new Alembic (relational schema) revision or a graph/vector data migration, or moving data between systems (relational DB import, memory export/import).
---

# Database migrations

cognee has **two migration chains**, run together:

| Chain | Changes | Lives in | Revision stored in |
|---|---|---|---|
| **Relational schema** (Alembic) | Tables and columns of the relational DB (users, datasets, ACLs, pipeline runs, …) | `cognee/alembic/` (`alembic.ini` is in `cognee/`) | `alembic_version` table |
| **Graph/vector data** | Cross-store data rewrites (re-keying node ids, adding graph columns) across graph DB, vector DB and relational ledger | `cognee/modules/migrations/` (`registry.py`, `versions/`) | Per dataset: `dataset_database.migration_revision` (access control on). Globally: `global_database_version.global_migration_revision` (access control off) |

## Use it

### They run by themselves

`run_migrations()` applies the relational chain first, then the data chain.
It runs:

- at API server startup (and in the Docker `entrypoint.sh` before gunicorn);
- on the first write in an SDK or CLI process (`remember`, `add`, `cognify`,
  `improve`, `memify`, memory imports), once per process;
- when you call `await cognee.run_migrations()`.

A fresh database is built by running the whole chain (no stamping). Steady
state costs one in-memory revision check. `ENABLE_AUTO_MIGRATIONS=false`
turns off all automatic runs; then run `cognee-cli upgrade` yourself.

Concurrent processes are serialized by a migration lock: a Postgres advisory
lock (works across hosts) or a file lock next to the SQLite DB (one host
only).

### Check and repair

```bash
cognee-cli current                   # stamped revision per database, and the last failure
cognee-cli history                   # the data chain, newest first
cognee-cli upgrade                   # relational to head, then data chain to head
cognee-cli upgrade <slug>            # data chain up to and including <slug>
cognee-cli upgrade --alembic <rev>   # pin the relational target
cognee-cli downgrade <slug|base> [--dataset UUID ...] [--alembic REV] [--force]
cognee-cli stamp <head|base|slug> [--dataset UUID ...] [--force]
```

- The positional revision is always a **data-chain slug**; relational
  targets go through `--alembic`.
- `downgrade` rewrites data and asks for confirmation. It only reverts
  spans where every migration defines a `down()`, and leaves the relational
  schema alone unless you pass `--alembic`.
- `stamp` changes only the stored data-chain revision, without running
  anything. Use `stamp base --dataset <id>` and then `upgrade` when a
  database's data drifted from its stamp (for example after restoring a
  backup); the chain is idempotent and converges it.
- `upgrade` runs even with `ENABLE_AUTO_MIGRATIONS=false`.

### A write is blocked

If a dataset's data migration failed, writes to that dataset are refused
until it succeeds (with access control off, any failure blocks all writes).
The server still starts. Run `cognee-cli current` to see the error, fix the
cause, then `cognee-cli upgrade`. A failed run is retried on the next start
or write.

### Moving data between systems (not schema migrations)

| Goal | Use |
|---|---|
| Turn an existing relational database into a graph | `migrate_relational_database(graph_db, schema)` (`cognee/tasks/ingestion/migrate_relational_database.py`), with the source DB set by `MIGRATION_DB_PROVIDER` / `_PATH` / `_NAME` / `_HOST` / `_PORT` / `_USERNAME` / `_PASSWORD`. Examples: `examples/demos/ingestion_and_migration/` |
| Export a dataset's memory | `await cognee.export(dataset, format=...)`: `"pydantic"` (in memory, default), or a file in `"cogx"`, `"json"`, `"graphml"`, `"cypher"` |
| Import from another memory system (Mem0, Zep/Graphiti, Letta, LangMem, a COGX archive) | Build a `MemorySource` (`cognee/modules/migration/sources/`) and pass it to `await cognee.remember(source, dataset_name=...)` |

There is no tool that moves a whole deployment from one database backend to
another.

## Pitfalls

- **Never regenerate `cognee/alembic/frozen_schema.py`.** It is the
  certified base schema the initial revision builds from, pinned by
  `cognee/tests/unit/test_frozen_schema_seal.py`. Schema changes ship as
  new revisions at head.
- **Never hand-type an Alembic revision id.** Hand-typed patterns
  (`a1b2c3d4e5f6`, …) already collided with a downstream chain that vendors
  this one (`b2c3d4e5f6a7`). Generate ids with `alembic revision`.
- **Never rename, remove, or reorder a data-chain entry.** The slug is what
  deployed databases store; an unknown stored slug disables the chain for
  that database.
- **A model change and its migration land together.** CI's "Migration/Model
  Lockstep Guard" fails otherwise.
- **The relational schema cannot be downgraded below the data-bookkeeping
  revisions** unless the data chain goes to `base` in the same call.
- `alembic.ini` is always the packaged one; `COGNEE_ALEMBIC_PATH` or
  `--alembic-path` only changes the scripts directory (for vendored chains).

## How it works

`run_migrations()` (`cognee/modules/migrations/startup.py`) takes the
migration lock, decides fresh vs existing (a `users` or `alembic_version`
table exists), runs Alembic in-process on a worker thread, then walks the
data chain per database with `runner.run_database_migrations`, stamping
after every step. After the chain, it syncs vector-adapter storage when the
recorded `cognee_version` differs from the library's
(`versions/adapter_storage_migration.py`, not a chain entry).

- Relational: `cognee/alembic.ini`, `cognee/alembic/env.py`,
  `cognee/alembic/versions/`, `cognee/alembic/frozen_schema.py`
- Data chain: `cognee/modules/migrations/` (`README.md` is the authoring
  contract; `registry.py`, `migration.py`, `runner.py`, `startup.py`,
  `versions/`)
- CLI: `cognee/cli/commands/migrate_command.py`

## Extending it

### A new Alembic revision

```bash
cd cognee                              # the directory with alembic.ini
uv run alembic revision -m "add foo to data"
```

The DB URL comes from the live relational engine, so your normal `.env`
settings apply. Follow the recent revisions (for example
`versions/e7f9a1c3d5b8_add_data_dataset_created_index.py`):

- **Idempotent and guarded:** inspect first (`sa.inspect(op.get_bind())`)
  and skip when the table is missing or the column/index already exists.
- **Branch on dialect** (`conn.dialect.name == "postgresql"`) for
  Postgres-only SQL. For Postgres enums use `postgresql.ENUM(...,
  create_type=False)` and create the type up front with `checkfirst`.
- **SQLite cannot drop or alter columns in place:** use
  `op.batch_alter_table(...)`.
- **Indexes:** plain `CREATE INDEX IF NOT EXISTS` inside the migration
  transaction, not `CONCURRENTLY` (which releases the version-row lock and
  lets concurrent workers into the same build). Repair invalid Postgres
  indexes via `pg_index.indisvalid`.
- A new model module outside the usual import path must be imported in
  `cognee/alembic/env.py`, or autogenerate will not see it.

### A new data migration

Read `cognee/modules/migrations/README.md` first. In short:

1. Write a module in `cognee/modules/migrations/versions/` with an `up`
   (and optionally `down`) function.
2. Append `Migration(slug=..., cognee_version=..., up=..., down_revision=<previous slug>, down=...)`
   to `MIGRATIONS` in `registry.py`. The chain is validated at import
   (linear, unique slugs).
3. Make it idempotent and cheap on empty stores, and crash-safe: re-key
   derived stores (vectors, ledger) first and rename in the graph last.
4. Freeze private copies of any logic you depend on; never import live
   models that may change later.

### Tests

- `cognee/tests/e2e/migrations/test_migration_model_lockstep.py` (+
  `schema_baseline.json`): the lockstep CI job.
- `cognee/tests/unit/test_run_migrations.py`: single Alembic head, startup
  behaviour.
- `cognee/tests/unit/test_frozen_schema_seal.py`: frozen schema fingerprint.
- `cognee/tests/unit/modules/migrations/`: data-chain unit tests.
- `cognee/tests/migrations/test_migration_lockstep.py`: data chain against
  real stores (seed, down, up, verify).
