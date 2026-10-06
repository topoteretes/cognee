# Running cognee on Turso (local)

Turso is a rewrite of SQLite in Rust that keeps the SQLite file format and SQL dialect and adds
multi-version concurrency control (MVCC) and vector distance functions. cognee can run all four
of its stores on it, as local database files:

| Layer | Provider setting | What it stores | Adapter |
|---|---|---|---|
| Relational | `DB_PROVIDER=turso` | users, datasets, pipeline runs, ACLs | `cognee/infrastructure/databases/relational/sqlalchemy/TursoAdapter.py` |
| Graph | `GRAPH_DATABASE_PROVIDER=turso` | graph-as-tables (`graph_node`, `graph_edge`) | `cognee/infrastructure/databases/graph/turso/adapter.py` |
| Vector | `VECTOR_DB_PROVIDER=turso` | one table per collection, `F32_BLOB` embeddings | `cognee/infrastructure/databases/vector/turso/TursoVectorAdapter.py` |
| Session cache | `CACHE_BACKEND=turso` | session Q&A, traces, context, usage logs (`cache.db`) | `cognee/infrastructure/databases/cache/sql/SqlCacheAdapter.py` |

The four layers are independent: any one of them can be Turso while the others stay on their defaults.
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
| `TURSO_JOURNAL_MODE` | `wal` | `wal`: SQLite-compatible write-ahead log, one writer at a time. `mvcc`: Turso's concurrent writes for the graph and vector stores (below); the relational DB and session cache always stay on `wal`. |
| `TURSO_BUSY_TIMEOUT_MS` | `120000` | how long a writer waits for the write lock in `wal` mode before `database is locked` |
| `TURSO_CONFLICT_RETRIES` | `5` | how many times the graph and vector write paths re-run a transaction that hit a write conflict |

### Runnable example

```bash
uv run python examples/guides/turso_local_example.py ingest    # add -> cognify -> search
uv run python examples/guides/turso_local_example.py verify    # fresh process: search the stored files
uv run python examples/guides/turso_local_example.py cleanup   # forget(everything=True) + remove files
```

The script sets the three providers itself and keeps its files under `.turso_example/` next to it.
`verify` runs in a new process and answers from the files `ingest` wrote, which is the persistence
check. `TURSO_JOURNAL_MODE=mvcc` runs the same flow with the graph and vector stores on concurrent
writes.

### Migrating from the previous `turso` provider

Before this version the `turso` provider ran on plain SQLite (`aiosqlite`) for the relational and
graph layers and on `libsql-experimental` for vectors. Existing local files need no migration: a
WAL-mode file written by stock SQLite opens on the Turso engine and vice versa (verified both
ways). The remote replica mode of the old vector and relational adapters is gone; configure a local
file or another provider. Setting `DB_TURSO_URL` / `DB_TURSO_AUTH_TOKEN` is now a hard error, and the
`turso` extra no longer installs `libsql-experimental`.

A Turso file can also be open in only one process at a time. The old aiosqlite-based relational
and graph providers allowed several processes on one file, so a deployment that ran the API server
next to a separate SDK/CLI process, or several server workers, on the same files breaks on upgrade.

## Transactions and concurrent writes

`wal` mode is SQLite's behaviour: writers take one lock, others wait up to `TURSO_BUSY_TIMEOUT_MS`
and then fail with `database is locked`. Files stay readable by stock SQLite. The driver's implicit
`BEGIN` only starts at the first INSERT/UPDATE/DELETE, so the graph and vector adapters open their
write transactions with `BEGIN IMMEDIATE` themselves (`write_transaction()`): a read-modify-write
(tag merges and removals, payload updates) holds the write lock from its first read, and a
concurrent writer waits and then reads its commit instead of being overwritten from a stale read.

`mvcc` mode (`PRAGMA journal_mode=mvcc`, applied on every connection) makes the **graph and
vector** adapters open write transactions with `BEGIN CONCURRENT`. Independent connections commit
in parallel; only two transactions that write the **same row** conflict, and the engine reports
that eagerly, at the statement, as `Write-write conflict`, aborting the later transaction. Every
graph and vector write is one self-contained transaction, which the adapters re-run
(`retry_on_conflict`, jittered backoff, `TURSO_CONFLICT_RETRIES` attempts).

The **relational DB and the session cache ignore `mvcc` and always run in `wal`**
(`TursoConfig.wal_only()`). Their writes go through ORM sessions whose callers run arbitrary code
between statements, so a conflicted transaction cannot be replayed: under `mvcc` the second of two
writers to one row (pipeline status, dataset rows, the session-context upsert) would fail and its
write would be lost. Under `wal` it waits on `busy_timeout` and succeeds. The price is write
throughput under heavy concurrency: in a local benchmark of concurrent session-cache writes in one
process, `wal` fell from about 1,200 to about 150 writes/s at 64 writers, while `mvcc` held about
1,200 on distinct rows but, without retries, lost most same-row writes.

Constraints of the mode:

- DDL needs a plain `BEGIN` ("DDL statements require an exclusive transaction"); cognee runs schema
  creation and migrations inside `exclusive_transaction()`, which switches the statement.
- The database file gains a `-log` companion and is no longer readable by stock `sqlite3`
  (cognee's dataset cleanup removes `-wal`, `-shm` and `-log`). Switching a file back to `wal`
  restores SQLite compatibility. Migration `c3d5e7f9a1b2`, which heals a standalone `cache.db`,
  opens a Turso cache through pyturso, so it also handles a cache file left in `mvcc` mode.
- Plain `BEGIN` writers still serialize with `database is locked`, so mixing tools that do not use
  `BEGIN CONCURRENT` gains nothing.
- MVCC is experimental upstream. The default stays `wal`.

Measured in `cognee/tests/e2e/turso/test_concurrent_writes.py` (both modes): 6 writers on
independent connections, 40 rows each, through the graph adapter, the vector adapter and raw
threads; a forced same-row conflict; retry recovery; and a reopen after the writes. The relational
adapter's tests pin that two same-row writers both commit with `TURSO_JOURNAL_MODE=mvcc` set.

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

# The same adapter tests on concurrent writes (CI runs both modes)
TURSO_JOURNAL_MODE=mvcc pytest <the paths above>

# Full search suite with every layer on Turso (needs LLM + embedding keys)
DB_PROVIDER=turso GRAPH_DATABASE_PROVIDER=turso VECTOR_DB_PROVIDER=turso CACHE_BACKEND=turso \
  pytest cognee/tests/test_search_db.py -v
```

Coverage: the adapter tests run in both `wal` and `mvcc` in CI. The end-to-end search suites run
in `wal` only, so `mvcc` is not covered end to end.

Every Turso test asserts `SELECT turso_version()` (or the `cognee_turso` driver name), so a test
that silently ran on SQLite would fail.

## Known limitations

| Limitation | Effect on cognee | Handling |
|---|---|---|
| No approximate vector index (`libsql_vector_idx` / `vector_top_k` are not supported) | similarity search is an exact `vector_distance_cos` scan | ~16–20 ms for top-15 over 20k × 384-dim rows on a laptop; fine for local datasets, not for millions of rows |
| No recursive CTEs (0.7.x) | k-hop neighborhoods, connected components | one query per hop; union-find in Python |
| No scalar subquery in `ON CONFLICT DO UPDATE SET` | vector upsert that merged `belongs_to_set` in SQL | tags are read and merged in Python, then a plain `excluded.payload` upsert |
| `json_each().value` returns a string element's raw JSON text, escapes included (`B\u00fccher` for `Bücher`); payloads are written with `json.dumps`, which escapes non-ASCII characters, quotes and backslashes | node set names in vector search filters and tag removal | names are compared on the decoded element (`payload ->> je.fullkey`); tag removal filters the names in Python inside one write transaction |
| Bind parameters must be `None`, numbers, `str` or `bytes` (no `register_adapter`) | raw `text()` statements binding `datetime`/`UUID` | SQLAlchemy-typed columns are unaffected; raw statements use a typed `bindparam` |
| Quoted identifiers are stored lowercased in `sqlite_master` | `has_collection` by exact name, PascalCase collection detection | case-insensitive lookup; collections detected by schema (`id, payload, vector`) |
| Parenthesized joins in a FROM clause (`JOIN (a JOIN b ON …)`) are rejected | SQLAlchemy emits them for joined-table inheritance (`User`/`Tenant` are `Principal` subclasses) | the dialect's compiler (`turso/compiler.py`) flattens the tree into a left-deep join chain |
| A join chain that combines several inner joins, a constant filter and a `LEFT JOIN` can drop a join condition and return extra rows; an affected query returns the wrong rows on every run | none known: no current cognee query has the shape, but users and permissions are read through joins on this engine | avoid the shape in new SQL, or check its rows against stock SQLite; `test_turso_engine_bugs.py` reproduces the smallest case and starts failing once a pyturso release fixes it |
| `VACUUM` requires an experimental engine flag | none (cognee never runs `VACUUM`) | – |
| MVCC files are unreadable by stock SQLite | external tooling on a live `mvcc` database | switch the file to `wal` first |
| One synchronous connection used from two threads at once aborts the process (Rust panic) | the vector adapter shares one connection | every call runs under `_connection_lock`; keep it that way |
| A database file can be open in one process at a time, in both journal modes. The engine holds the file lock until that process exits (closing its connections does not release it), and it also blocks stock `sqlite3` | the API server and a separate SDK/CLI process, or several server workers, cannot share one Turso deployment; the stock SQLite backends allow this. cognee raises `TursoDatabaseInUseError`, naming the file, instead of the engine's `Locking error` | run one cognee process per set of Turso files (the server's default `gunicorn -w 1` is fine), or give each process its own `SYSTEM_ROOT_DIRECTORY` |

## Upstream findings

Each finding will be filed upstream with a minimal reproduction and linked from SDK-664. When a
finding is fixed in a released `pyturso`, the matching workaround (noted in the table above) can
be retired.

1. `sqlite+aioturso` dialect fails on SQLAlchemy 2.0.4x+: `'AsyncAdapt_turso_dbapi' object has no
   attribute 'has_stop'`. cognee's dialect sets it.
2. The dialect's reflection mixin returns `[]` for indexes, foreign keys and unique constraints
   although `PRAGMA index_list` / `foreign_key_list` work; with it, cognee's idempotent migrations
   collide on existing indexes and Alembic batch operations would drop constraints. cognee's
   dialect restores stock SQLite reflection. Fixed upstream in the 0.8.0 release candidates.
3. Scalar subquery in `ON CONFLICT DO UPDATE SET`: `Parse error: Subquery is not supported in this
   position` (0.7.2 and 0.8.0rc11).
4. An outer `json_each()` over an unaliased column is misresolved inside `UPDATE`/`DELETE`: on
   0.7.2 the tag-removal `UPDATE ... json_each(payload, ...)` runs without error but changes no rows
   (stock SQLite strips the tag), and qualifying the column with a quoted mixed-case table name fails
   with `no such table` (finding 6). Aliasing the table (`AS t`, `json_each(t.payload, ...)`) works.
   Earlier shapes failed with `bind index 1 is out of bounds`; 0.8.0rc11 reports `'json_each' is not
   a function`.
5. `WITH RECURSIVE`: `Recursive CTEs are not yet supported` (0.7.2); 0.8.0rc11 accepts the syntax
   but aborts the process (`Fatal Python error: Abort`) on cognee's neighborhood query.
6. Quoted identifiers lowercased in `sqlite_master` (stock SQLite preserves case).
7. Only primitive bind parameter types; no `register_adapter` equivalent.
8. `JOIN (a JOIN b ON …)`: `Parenthesized FROM clause subqueries are not supported`, although
   the engine reports SQLite 3.50 (SQLite has accepted the form since 3.7.16).
9. `json_each().value` returns a string element's raw JSON text instead of the decoded string
   (0.7.2): for the stored array `["B\u00fccher"]` (how `json.dumps` writes `["Bücher"]`) it
   returns the escape text `B\u00fccher` itself, where stock SQLite returns `Bücher`.
   `payload ->> je.fullkey` and `json_extract(payload, je.fullkey)` decode correctly.
10. A join condition is dropped (0.7.2): with `u=(2,1)`, `a=(1,1)`, `p=(3,1)` and `b` empty
    (`id INTEGER PRIMARY KEY, k INTEGER`), `SELECT u.id, a.id FROM u JOIN a ON u.id = a.id AND
    a.k = 1 JOIN p ON p.k = a.k LEFT JOIN b ON u.k = b.k` returns `(2, 1)`; stock SQLite returns
    no rows. Removing the constant filter, the third inner table or the `LEFT JOIN` gives the
    right result; moving the conditions to `WHERE` does not.
