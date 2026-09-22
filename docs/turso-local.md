# Running cognee on Turso (local)

Turso is a rewrite of SQLite in Rust that keeps the SQLite file format and SQL dialect and adds
multi-version concurrency control (MVCC) and vector distance functions. cognee can run all three
of its stores on it, as local database files:

| Layer | Provider setting | What it stores | Adapter |
|---|---|---|---|
| Relational | `DB_PROVIDER=turso` | users, datasets, pipeline runs, ACLs | `cognee/infrastructure/databases/relational/sqlalchemy/TursoAdapter.py` |
| Graph | `GRAPH_DATABASE_PROVIDER=turso` | graph-as-tables (`graph_node`, `graph_edge`) | `cognee/infrastructure/databases/graph/turso/adapter.py` |
| Vector | `VECTOR_DB_PROVIDER=turso` | one table per collection, `F32_BLOB` embeddings | `cognee/infrastructure/databases/vector/turso/TursoVectorAdapter.py` |
| Session cache | `CACHE_BACKEND=turso` | session Q&A, traces, context, usage logs (`cache.db`) | `cognee/infrastructure/databases/cache/sql/SqlCacheAdapter.py` |

The layers are independent: any one of them can be Turso while the others stay on their defaults.
The session cache mirrors the `sqlite` backend exactly (a `cache.db` next to the relational
database, same tables and upserts) and only swaps the driver; it is not switched automatically
with `DB_PROVIDER`, so set `CACHE_BACKEND=turso` explicitly, as with `postgres`.
All three share `cognee/infrastructure/databases/turso/`: the SQLAlchemy dialect
(`sqlite+cognee_turso://`), the `TURSO_*` settings, the transaction/retry policy and the
file-cleanup rule.

## Versions

| Component | Version | Notes |
|---|---|---|
| `pyturso` (import name `turso`) | `>=0.7.2,<0.8` (pinned in the `turso` extra) | the Turso rewrite engine and its Python bindings; `SELECT turso_version()` proves it is the executing engine |
| SQL dialect tracked by the engine | SQLite 3.50.4 | `SELECT sqlite_version()` |
| SQLAlchemy | 2.0.x (cognee's pin) | the bundled `sqlite+aioturso` dialect fails on 2.0.4x+, see "Upstream findings" |
| Python | 3.10 – 3.14 | wheels for Linux x86_64/aarch64, macOS, Windows |

`pyturso` 0.8.0 release candidates exist. They add recursive CTEs, but 0.8.0rc11 aborts the
process on the neighborhood query cognee used, so cognee stays on 0.7.x and avoids recursive CTEs.

## Setup

```bash
pip install cognee"[turso]"          # or: uv pip install -e ".[turso]"

# .env — pick any subset
DB_PROVIDER=turso
GRAPH_DATABASE_PROVIDER=turso
VECTOR_DB_PROVIDER=turso
CACHE_BACKEND=turso
LLM_API_KEY=...                       # cognify needs an LLM as usual
```

Files land under the system databases directory (`SYSTEM_ROOT_DIRECTORY/databases`): the
relational database as `DB_NAME`, and with access control on (the default) one graph file
(`graph_<dataset_id>.db`) and one vector file (`<dataset_id>.turso.db`) per dataset. Paths can be
pinned with `GRAPH_DATABASE_URL` / `VECTOR_DB_URL` (absolute paths). Remote Turso databases are
**not** supported in this version: a set `DB_TURSO_URL`, `GRAPH_DATABASE_KEY` or a `libsql://` /
`https://` vector URL is a hard error, never a silent fallback to a local file.

Engine-wide settings (`TursoConfig`, env prefix `TURSO_`):

| Setting | Default | Meaning |
|---|---|---|
| `TURSO_JOURNAL_MODE` | `wal` | `wal`: SQLite-compatible write-ahead log, one writer at a time. `mvcc`: Turso's concurrent writes (below). |
| `TURSO_BUSY_TIMEOUT_MS` | `120000` | how long a writer waits for the write lock in `wal` mode before `database is locked` |
| `TURSO_CONFLICT_RETRIES` | `5` | how many times cognee's own write paths re-run a transaction that hit a write conflict |

### Runnable example

```bash
uv run python examples/guides/turso_local_example.py ingest    # add -> cognify -> search
uv run python examples/guides/turso_local_example.py verify    # fresh process: search the stored files
uv run python examples/guides/turso_local_example.py cleanup   # forget(everything=True) + remove files
```

The script sets the three providers itself and keeps its files under `.turso_example/` next to it.
`verify` runs in a new process and answers from the files `ingest` wrote, which is the persistence
check. `TURSO_JOURNAL_MODE=mvcc` runs the same flow on concurrent writes.

### Migrating from the previous `turso` provider

Before this version the `turso` provider ran on plain SQLite (`aiosqlite`) for the relational and
graph layers and on `libsql-experimental` for vectors. Existing local files need no migration: a
WAL-mode file written by stock SQLite opens on the Turso engine and vice versa (verified both
ways). The remote replica mode of the old vector and relational adapters is gone; configure a local
file or another provider.

## Transactions and concurrent writes

`wal` mode is SQLite's behaviour: writers take one lock, others wait up to `TURSO_BUSY_TIMEOUT_MS`
and then fail with `database is locked`. Files stay readable by stock SQLite.

`mvcc` mode (`PRAGMA journal_mode=mvcc`, applied on every connection) makes cognee open write
transactions with `BEGIN CONCURRENT`. Independent connections commit in parallel; only two
transactions that write the **same row** conflict, and the engine reports that eagerly, at the
statement, as `Write-write conflict`, aborting the later transaction. cognee's adapters re-run the
whole transaction (`retry_on_conflict`, jittered backoff, `TURSO_CONFLICT_RETRIES` attempts).
Constraints of the mode:

- DDL needs a plain `BEGIN` ("DDL statements require an exclusive transaction"); cognee runs schema
  creation and migrations inside `exclusive_transaction()`, which switches the statement.
- The database file gains a `-log` companion and is no longer readable by stock `sqlite3`
  (cognee's dataset cleanup removes `-wal`, `-shm` and `-log`). Switching a file back to `wal`
  restores SQLite compatibility. This also means migration `c3d5e7f9a1b2`, which heals a
  standalone `cache.db` through stdlib `sqlite3`, only works on a `wal`-mode cache file.
- Plain `BEGIN` writers still serialize with `database is locked`, so mixing tools that do not use
  `BEGIN CONCURRENT` gains nothing.
- MVCC is experimental upstream. The default stays `wal`.

Measured in `cognee/tests/e2e/turso/test_concurrent_writes.py` (both modes): 6 writers on
independent connections, 40 rows each, through the graph adapter, the vector adapter and raw
threads; a forced same-row conflict; retry recovery; and a reopen after the writes.

## Test commands

```bash
# Adapter-level, no LLM needed (also the CI job "Turso Tests")
pytest cognee/tests/unit/infrastructure/databases/turso_backend \
       cognee/tests/unit/infrastructure/databases/relational/test_turso_adapter.py \
       cognee/tests/unit/infrastructure/databases/relational/test_create_relational_engine.py \
       cognee/tests/unit/infrastructure/databases/graph/test_turso_graph_dataset_database_handler.py \
       cognee/tests/unit/infrastructure/databases/vector/test_turso_adapter.py \
       cognee/tests/unit/infrastructure/databases/cache/test_sql_cache_turso.py \
       cognee/tests/unit/infrastructure/databases/cache/test_sql_adapter_crud.py \
       cognee/tests/e2e/turso/test_turso_adapter.py \
       cognee/tests/e2e/turso/test_concurrent_writes.py

# Full search suite with every layer on Turso (needs LLM + embedding keys)
DB_PROVIDER=turso GRAPH_DATABASE_PROVIDER=turso VECTOR_DB_PROVIDER=turso CACHE_BACKEND=turso \
  pytest cognee/tests/test_search_db.py -v
```

Every Turso test asserts `SELECT turso_version()` (or the `cognee_turso` driver name), so a test
that silently ran on SQLite would fail.

## Known limitations

| Limitation | Effect on cognee | Handling |
|---|---|---|
| No approximate vector index (`libsql_vector_idx` / `vector_top_k` are not supported) | similarity search is an exact `vector_distance_cos` scan | ~16–20 ms for top-15 over 20k × 384-dim rows on a laptop; fine for local datasets, not for millions of rows |
| No recursive CTEs (0.7.x) | k-hop neighborhoods, connected components | one query per hop; union-find in Python |
| No scalar subquery in `ON CONFLICT DO UPDATE SET` | vector upsert that merged `belongs_to_set` in SQL | tags are read and merged in Python, then a plain `excluded.payload` upsert |
| No bind parameter inside a nested `json_each(?)` subquery | vector tag removal | payloads filtered in Python, written back with plain binds |
| Bind parameters must be `None`, numbers, `str` or `bytes` (no `register_adapter`) | raw `text()` statements binding `datetime`/`UUID` | SQLAlchemy-typed columns are unaffected; raw statements use a typed `bindparam` |
| Quoted identifiers are stored lowercased in `sqlite_master` | `has_collection` by exact name, PascalCase collection detection | case-insensitive lookup; collections detected by schema (`id, payload, vector`) |
| Parenthesized joins in a FROM clause (`JOIN (a JOIN b ON …)`) are rejected | SQLAlchemy emits them for joined-table inheritance (`User`/`Tenant` are `Principal` subclasses) | the dialect's compiler (`turso/compiler.py`) flattens the tree into a left-deep join chain |
| `VACUUM` requires an experimental engine flag | none (cognee never runs `VACUUM`) | – |
| MVCC files are unreadable by stock SQLite | external tooling on a live `mvcc` database | switch the file to `wal` first |
| One synchronous connection used from two threads at once aborts the process (Rust panic) | the vector adapter shares one connection | every call runs under `_connection_lock`; keep it that way |

## Upstream findings

Each of these has a minimal reproduction in `cognee/tests/e2e/turso/turso_compat_repros.py`
(run it directly to print the current status against the installed `pyturso`).

1. `sqlite+aioturso` dialect fails on SQLAlchemy 2.0.4x+: `'AsyncAdapt_turso_dbapi' object has no
   attribute 'has_stop'`. cognee's dialect sets it.
2. The dialect's reflection mixin returns `[]` for indexes, foreign keys and unique constraints
   although `PRAGMA index_list` / `foreign_key_list` work; with it, cognee's idempotent migrations
   collide on existing indexes and Alembic batch operations would drop constraints. cognee's
   dialect restores stock SQLite reflection. Fixed upstream in the 0.8.0 release candidates.
3. Scalar subquery in `ON CONFLICT DO UPDATE SET`: `Parse error: Subquery is not supported in this
   position` (0.7.2 and 0.8.0rc11).
4. Bind parameter inside a nested `json_each(?)` subquery: `bind index 1 is out of bounds` (0.7.2),
   `'json_each' is not a function` (0.8.0rc11).
5. `WITH RECURSIVE`: `Recursive CTEs are not yet supported` (0.7.2); 0.8.0rc11 accepts the syntax
   but aborts the process (`Fatal Python error: Abort`) on cognee's neighborhood query.
6. Quoted identifiers lowercased in `sqlite_master` (stock SQLite preserves case).
7. Only primitive bind parameter types; no `register_adapter` equivalent.
8. `JOIN (a JOIN b ON …)`: `Parenthesized FROM clause subqueries are not supported`, although
   the engine reports SQLite 3.50 (SQLite has accepted the form since 3.7.16).
