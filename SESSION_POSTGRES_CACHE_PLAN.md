# Implementation Plan: Postgres-backed Session Store (CacheDBInterface adapter)

**Goal:** store cognee session data (QA entries, agent traces, usage logs, and small string KV values) in Postgres via a new `CacheDBInterface` adapter, removing the Redis requirement for session memory. Default backend stays `fs`; Redis and Tapes keep working unchanged. The design keeps a clean seam for a future Turbopuffer backend.

**Design stance:** minimal-surface adapter that slots into the existing backend plug point (like `FSCacheAdapter`), follows the graph Postgres adapter's engine/schema patterns, and adds **zero new dependencies** (SQLAlchemy is core; asyncpg/psycopg2 ship in the existing `postgres` extra). Where the minimal approach was naive (TTL purging, multi-worker RMW, lock semantics, prune scope), the robust alternatives are grafted in below — each divergence from the minimal design is justified inline.

---

## 1. Overview & goals

- Add `CACHE_BACKEND=postgres` as a fourth cache backend alongside `redis`, `fs`, `tapes`.
- Implement `PostgresCacheAdapter(CacheDBInterface)` with full behavioral parity to `RedisAdapter`/`FSCacheAdapter` (TTL semantics, merge semantics, error contracts), improving on known Redis races where doing so is observably identical (atomic deletes, `FOR UPDATE` updates).
- Support the formal string KV methods on `CacheDBInterface` (`get_value`/`set_value`/`delete_value`) for small cache values. The old graph-to-session sync contract is removed; new backends should not add an `async_redis` shim for it.
- Non-goals (v1): data migration tooling (cache is 7-day-TTL ephemeral), per-dataset DB isolation (sessions are per-user), distributed `SHARED_LADYBUG_LOCK` support on Postgres (Phase 6), multi-worker `improve()` mutex.

---

## 2. Current state

**Cache interface** — `cognee/infrastructure/databases/cache/cache_db_interface.py`: ABC with sync lock methods (`acquire_lock`/`release_lock`, concrete `hold_lock()` contextmanager), async QA CRUD (`create_qa_entry`, `get_latest_qa_entries`, `get_all_qa_entries`, `update_qa_entry`, `delete_feedback`, `delete_qa_entry`, `delete_session`), agent traces (`append_agent_trace_step`, `get_agent_trace_session`, `get_agent_trace_feedback`, `get_agent_trace_count`), `prune`, `close`, `log_usage`/`get_usage_logs`, plus concrete back-compat shims `add_qa`/`get_latest_qa`/`get_all_qas` (do not reimplement). Base `__init__(host, port, lock_key="default_lock", log_key="usage_logs")`. Payload models: `SessionQAEntry`/`SessionAgentTraceEntry` in `cognee/infrastructure/databases/cache/models.py` (validators: feedback_score 1–5, `used_graph_element_ids` only `node_ids`/`edge_ids`, `memify_metadata` str→bool, trace sanitization/truncation).

**Backends** — `cache/redis/RedisAdapter.py` (sync `redis.Redis` for locks + async `redis.asyncio.Redis` for data; one Redis LIST per key `agent_sessions:{user_id}:{session_id}` / `agent_traces:{user_id}:{session_id}` / `{log_key}:{user_id}`; JSON-string elements; `EXPIRE`-on-write sliding TTL; `FLUSHDB` prune); `cache/fscache/FsCacheAdapter.py` (diskcache, whole list as one JSON value under `self.cache`, `cache.transact()`, lock methods raise `SharedLadybugLockRequiresRedisError`); `cache/tapes/TapesCacheAdapter.py` (FS subclass mirroring QA creates over HTTP).

**Factory** — `cache/get_cache_engine.py`: `@lru_cache`'d `create_cache_engine(...)` branches on `CacheConfig.cache_backend` (`Literal["redis","fs","tapes"]`, default `"fs"`, in `cache/config.py`); returns `None` when `caching` and `usage_logging` are both off; `close_cache_engine()` closes + `cache_clear()`s. **Verified gotcha:** `from ...RedisAdapter import RedisAdapter` executes unconditionally inside the caching-on block, *before* the backend dispatch — it only works because the unused core dep `fakeredis[lua]` transitively installs `redis`.

**Session manager** — `cognee/infrastructure/session/session_manager.py` (built by `get_session_manager.py`): wraps every interface method, no-ops when `cache_engine is None`. It no longer stores graph snapshots in session prompts. For one release, `delete_session` still deletes the legacy `graph_knowledge:{user_id}:{session_id}` key via `delete_value` so old cache rows do not linger.

**Locks** — two unrelated systems: (1) `cognee/infrastructure/locks/session_lock.py`: pure in-process asyncio locks (per-`(session_id, op)` dict + improve-lock set), explicitly single-worker scope, **no Redis dependency, no change required**; (2) `CacheDBInterface.acquire_lock`/`release_lock`: **sync** methods used only by the Ladybug graph adapter (`graph/ladybug/adapter.py` ~line 187, via `asyncio.to_thread`) when `SHARED_LADYBUG_LOCK=true` — Redis-only today.

**Relational sibling** — `cognee/modules/session_lifecycle/models.py` (`session_records`, `session_model_usage`): alembic-managed lifecycle/metrics rows, already Postgres-compatible, untouched by this plan. No FK between cache rows and `session_records` (cache rows TTL-expire; lifecycle rows persist — by design).

**Template** — `cognee/infrastructure/databases/graph/postgres/adapter.py` + `tables.py`: own `create_async_engine(uri, json_serializer=lambda obj: json.dumps(obj, cls=JSONEncoder), **pool_args)` (verified, lines 40–58), `async_sessionmaker(expire_on_commit=False)`, private `MetaData()` with `create_all(checkfirst=True)` in `initialize()` — not alembic.

---

## 3. Target architecture

```
SessionManager / usage_logger / forget / prune_system / memify tasks
        │  (unchanged call sites)
        ▼
get_cache_engine()  ──reads──  CacheConfig (.env: CACHE_BACKEND, CACHE_DB_URL, ...)
        │ @lru_cache create_cache_engine()
        ├── "redis"    → RedisAdapter            (unchanged)
        ├── "fs"       → FSCacheAdapter          (unchanged)
        ├── "tapes"    → TapesCacheAdapter       (unchanged)
        └── "postgres" → PostgresCacheAdapter    (NEW, lazy import)
                            │
                            ├── own async engine (postgresql+asyncpg://...)
                            │     json_serializer=JSONEncoder  (UUID/datetime-safe JSONB)
                            ├── tables (private MetaData, create-on-init):
                            │     cache_qa_entries   ← agent_sessions:{u}:{s} lists
                            │     cache_trace_entries← agent_traces:{u}:{s} lists
                            │     cache_usage_logs   ← {log_key}:{u} lists
                            │     cache_kv           ← generic small string KV values
                            └── get_value/set_value/delete_value over cache_kv
```

---

## 4. Postgres adapter design

### 4.1 Module layout

Mirror the existing backend layout:

```
cognee/infrastructure/databases/cache/postgres/
├── __init__.py                  # re-export PostgresCacheAdapter
├── tables.py                    # private MetaData + SQLAlchemy Core tables
└── PostgresCacheAdapter.py      # class PostgresCacheAdapter(CacheDBInterface)
```

### 4.2 Class skeleton

```python
class PostgresCacheAdapter(CacheDBInterface):
    def __init__(
        self,
        connection_string: str,                  # any SQLAlchemy async URL (asyncpg in prod, aiosqlite in tests)
        lock_key: str = "default_lock",
        log_key: str = "usage_logs",
        session_ttl_seconds: int | None = 604800,
        agentic_lock_expire: int = 240,          # stored now, used by Phase 6 advisory locks
        agentic_lock_timeout: int = 300,
        purge_interval_seconds: int = 900,
    ):
        super().__init__(host="", port=0, lock_key=lock_key, log_key=log_key)
        self.db_uri = connection_string
        self.session_ttl_seconds = session_ttl_seconds
        pool_args = dict(get_relational_config().pool_args or {})
        self.engine = create_async_engine(
            connection_string,
            json_serializer=lambda obj: json.dumps(obj, cls=JSONEncoder),
            **pool_args,
        )
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._last_purge = 0.0
```

Notes:
- Call `super().__init__` (Redis does; FS doesn't — calling it gives `lock_key`/`log_key`/`lock` attributes).
- Single hashable `connection_string` arg keeps the `@lru_cache`'d factory happy.
- Owning the engine (rather than borrowing `get_relational_engine()`'s) keeps `close()` safe: `close_cache_engine()` → `adapter.close()` → `engine.dispose()` must not kill the shared relational pool.
- Engine-level `json_serializer` with `cognee.modules.storage.utils.JSONEncoder` is load-bearing: without it, asyncpg JSONB inserts containing UUIDs/datetimes fail (graph-adapter gotcha, replicated).
- Connection validation is **lazy** (in `_ensure_initialized`), not eager like Redis's `ping()` — the constructor must stay sync and lru_cache-friendly; the contract only requires `CacheConnectionError` to surface, which it does on first use. Also raise a clear `CacheConnectionError("CACHE_BACKEND=postgres requires cognee[postgres]")` if the asyncpg import fails.

### 4.3 Table DDL (`cache/postgres/tables.py`)

Private `MetaData()` — **not** the relational declarative `Base`, **not** alembic-managed (graph-adapter precedent: `graph_node`/`graph_edge` are create-on-init while `session_records` is alembic-managed; private MetaData keeps alembic autogenerate from seeing these). Payload columns use `JSONB().with_variant(JSON(), "sqlite")` so unit tests run on aiosqlite. `cache_`-prefixed names avoid collision with `session_records`/`session_model_usage`.

```sql
-- QA entries: one row per entry; qa_id promoted to a column for direct UPDATE
-- (Redis buries it in JSON and does linear scan + LSET)
CREATE TABLE cache_qa_entries (
    seq         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- insertion order
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    qa_id       TEXT NOT NULL,
    payload     JSONB NOT NULL,            -- full SessionQAEntry.model_dump() (incl. qa_id, time)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NULL,          -- NULL = no expiry
    UNIQUE (user_id, session_id, qa_id)
);
CREATE INDEX idx_cache_qa_session  ON cache_qa_entries (user_id, session_id, seq);
CREATE INDEX idx_cache_qa_expires  ON cache_qa_entries (expires_at) WHERE expires_at IS NOT NULL;

-- Agent traces: append-only
CREATE TABLE cache_trace_entries (
    seq         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    payload     JSONB NOT NULL,            -- SessionAgentTraceEntry.model_dump()
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NULL
);
CREATE INDEX idx_cache_trace_session ON cache_trace_entries (user_id, session_id, seq);
CREATE INDEX idx_cache_trace_expires ON cache_trace_entries (expires_at) WHERE expires_at IS NOT NULL;

-- Usage logs (Redis key {log_key}:{user_id})
CREATE TABLE cache_usage_logs (
    seq         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    log_key     TEXT NOT NULL,             -- adapter's self.log_key
    user_id     TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NULL
);
CREATE INDEX idx_cache_usage ON cache_usage_logs (log_key, user_id, seq);

-- String KV for small cache values; legacy graph_knowledge:{u}:{s} keys may be deleted by SessionManager
CREATE TABLE cache_kv (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    expires_at  TIMESTAMPTZ NULL
);
```

Notes:
- `session_id` stays a **plain string, no FK** to `session_records` (same `session_id` exists under multiple users; cache rows expire while lifecycle rows persist). Keying is always `(user_id, session_id)` — the exact relational analogue of the Redis key prefixes, which also preserves cross-user reads (recall / sessions router resolve the *owner's* `user_id` from `SessionRecord`).
- JSONB, not bytea/pickle: parity with `graph_node.properties` and the Redis JSON-string encoding; queryable for debugging.
- Hot-path: `get_latest_qa_entries` runs on every completion (last 10); `idx_cache_qa_session` makes the tail read an index scan. Rows can be tens of KB (payload embeds `context`) — a later `payload - 'context'` projection optimization is possible, but the contract returns full entries today; don't change behavior.

### 4.4 Method-by-method implementation

Every public async method: `await self._ensure_initialized()` first (once-flag under `self._init_lock`; runs `conn.run_sync(cache_metadata.create_all, checkfirst=True)`, wraps first-connect failure in `CacheConnectionError`), then `async with self.sessionmaker() as session, session.begin():`. Error contract (pinned by `test_redis_adapter_crud.py`/`test_fs_adapter_crud.py`): backend/SQLAlchemy errors → wrap in `CacheConnectionError`; create-path *validation* failures → also `CacheConnectionError` (odd but tested); update-path validation → `SessionQAEntryValidationError` **propagated unwrapped**.

| Method | Implementation |
|---|---|
| `create_qa_entry(user_id, session_id, question, context, answer, qa_id=None, ...)` | `qa_id: str \| None = None` with `str(uuid4())` fallback — match the adapters, not the ABC's `qa_id: str`. Build `SessionQAEntry` (stamps `time=datetime.utcnow().isoformat()`, runs validators; failure → `CacheConnectionError`). `INSERT` `payload=entry.model_dump()`; session-wide TTL refresh (§4.5); scoped lazy purge. One transaction. |
| `get_latest_qa_entries(u, s, last_n=5)` | `SELECT payload WHERE u/s AND not-expired ORDER BY seq DESC LIMIT :n`, reverse in Python → chronological; `SessionQAEntry.model_validate` each. **Return `[]` always on empty, including `last_n==1`** — the Redis `None`-on-`last_n==1` quirk is not replicated; all callers go through `SessionManager` and treat falsy uniformly (FS already returns `[]`). Pin in the unit test. |
| `get_all_qa_entries(u, s)` | Same, `ORDER BY seq ASC`, no LIMIT; `[]` if none. |
| `update_qa_entry(u, s, qa_id, ...)` | One transaction: `SELECT payload ... WHERE user_id/session_id/qa_id ... FOR UPDATE` (`with_for_update()`; no-op on the sqlite test variant, fine). No row → `False`. Merge in Python with exact existing semantics: `None` preserves every field; `memify_metadata = {**existing, **new}` (MERGE not replace — `apply_feedback_weights`/`apply_frequency_weights` idempotency flags depend on it); re-validate via `SessionQAEntry.model_validate`, letting `SessionQAEntryValidationError` propagate; `UPDATE payload`; TTL refresh; `True`. `FOR UPDATE` makes this RMW multi-worker-safe — strictly better than Redis's load/LSET race, observably identical. |
| `delete_feedback(u, s, qa_id)` | Same `FOR UPDATE` pattern; set `feedback_text`/`feedback_score` to `None` in payload; TTL refresh; bool. (This is the only way to clear feedback — `update_qa_entry(feedback_score=None)` preserves.) |
| `delete_qa_entry(u, s, qa_id)` | Single atomic `DELETE ... WHERE user_id/session_id/qa_id` (replaces Redis's non-atomic DEL+RPUSH-loop rewrite — crash-safe improvement); TTL refresh on remaining rows (Redis re-applies TTL after rewrite); return `rowcount > 0`. No "drop key when empty" step — empty session ≡ zero rows. |
| `delete_session(u, s)` | One transaction: `DELETE FROM cache_qa_entries` + `DELETE FROM cache_trace_entries` for `(u, s)`; `True` if either rowcount > 0. (`SessionManager.delete_session` separately deletes the legacy `graph_knowledge:` key via `delete_value` — keep that division for one release.) |
| `append_agent_trace_step(u, s, trace_id, origin_function, status, ...)` | Build `SessionAgentTraceEntry` (`method_params or {}`; model validators do sanitization/truncation for free; blank `trace_id`/`origin_function` → `CacheConnectionError`); INSERT; trace-session TTL refresh. |
| `get_agent_trace_session(u, s, last_n=None)` | `ORDER BY seq ASC`; when `last_n is not None`, `DESC LIMIT` + reverse. Reads do NOT refresh TTL. |
| `get_agent_trace_feedback(u, s, last_n=None)` | `[e.session_feedback for e in await self.get_agent_trace_session(...)]` — exactly the delegation both adapters use. |
| `get_agent_trace_count(u, s)` | `SELECT count(*)` with expiry filter; 0 for missing session. |
| `prune()` | `DELETE FROM` / `TRUNCATE` the **four cache tables only**. Deliberate, documented divergence from Redis `FLUSHDB` (which nukes co-tenant keys and other users' data) — strictly safer; both proposals agree. |
| `close()` | `await self.engine.dispose(close=True)`; dispose the sync lock engine if created (Phase 6); idempotent, swallow + debug-log errors (called by `close_cache_engine` and `prune_data.py`). Reset `self._initialized = False` so a reused instance re-inits. |
| `log_usage(user_id, log_entry, ttl=604800)` | INSERT into `cache_usage_logs` with `log_key=self.log_key`, `expires_at = now()+ttl` if ttl else NULL; **also refresh `expires_at` on the user's existing rows** under the same `(log_key, user_id)` — Redis `EXPIRE`s the whole list, so per-row-only TTL would diverge. |
| `get_usage_logs(user_id, limit=100)` | `SELECT payload WHERE log_key/user_id AND not-expired ORDER BY seq DESC LIMIT :limit` — already most-recent-first (Redis does lrange + reverse). |
| `acquire_lock()` / `release_lock(lock=None)` | **SYNC — do not make async** (Ladybug calls them via `asyncio.to_thread`). **v1: raise `SharedLadybugLockRequiresRedisError`**, imported from `cognee.infrastructure.databases.exceptions.exceptions` directly (it is NOT re-exported in the package `__init__`) — exactly like `FsCacheAdapter`. Inherited `hold_lock()` then raises too — correct. Phase 6 upgrade in §7. |
| Inherited (do not reimplement) | `hold_lock()`, `add_qa`, `get_latest_qa`, `get_all_qas`. |

**Concurrency posture:** unlike the graph adapter, **no global `asyncio.Lock` serializing writes** — row-level `FOR UPDATE` + single-statement deletes give correct multi-worker behavior with better throughput; the graph adapter's lock exists for bulk upsert batches that don't occur here. Add the graph adapter's tenacity retry on asyncpg `DeadlockDetectedError` as belt-and-braces.

### 4.5 TTL strategy: `expires_at` + read-time filter + lazy purge + throttled sweep

Replicates Redis sliding whole-key TTL exactly:

- **Write refresh:** every mutating op runs, inside its transaction, `UPDATE <table> SET expires_at = now() + :ttl WHERE user_id=:u AND session_id=:s` (skipped when `session_ttl_seconds` is `None`/`<= 0` — `expires_at` stays NULL, expiry disabled). Whole-session refresh matches `EXPIRE session_key`. Sessions are small (tens–hundreds of rows); cheap.
- **Read filter:** every SELECT adds `AND (expires_at IS NULL OR expires_at > now())`. Reads never refresh TTL (tested Redis behavior). Correctness never depends on purging.
- **Purge** ("purge on init only" would leak rows in long-lived processes): (a) scoped `DELETE ... WHERE user_id=:u AND session_id=:s AND expires_at <= now()` opportunistically on writes to that session; (b) a throttled global sweep `DELETE FROM <each table> WHERE expires_at <= now()` at most once per `purge_interval_seconds` (default 900) per process, guarded by `pg_try_advisory_lock` on Postgres (skip the guard on the sqlite test variant) so concurrent workers don't stampede. No external cron. Precedents: `FSCacheAdapter.cache.expire()` on init; `session_lifecycle` computing "abandoned" at read time.
- **KV behavior:** string KV values are exact-key values with optional TTL support where the adapter exposes it. No graph-to-session checkpoint keys are written anymore.

### 4.6 String KV methods

The old hidden `async_redis` duck-type contract is gone with graph-to-session sync. Implement the formal `CacheDBInterface` string KV methods directly:

```python
async def get_value(self, key: str) -> str | None: ...
async def set_value(self, key: str, value: str, ttl: int | None = None) -> None: ...
async def delete_value(self, key: str) -> None: ...
```

`get_value` returns `str | None`. `set_value` should upsert by key and set `expires_at` when a TTL is supplied. `delete_value` should be idempotent.

---

## 5. Config & provider selection changes

**`cognee/infrastructure/databases/cache/config.py`:**
- `cache_backend: Literal["redis", "fs", "tapes", "postgres"] = "fs"` (default **unchanged**).
- New fields: `cache_db_url: Optional[str] = None` (env `CACHE_DB_URL`, e.g. `postgresql+asyncpg://cognee:cognee@localhost:5432/cognee_db`) and `cache_purge_interval_seconds: int = 900`. **Single-URL chosen over five discrete `CACHE_DB_*` fields**: one hashable kwarg through the lru_cache'd factory, one config field instead of five, and it matches the adapter's `connection_string` constructor. Do **not** reuse `cache_host`/`cache_port` — they default to Redis's `localhost:6379` (silent misdirection risk).
- Fallback resolution (helper `_resolve_cache_db_url` in `get_cache_engine.py`, mirroring `get_graph_engine.py`'s GRAPH_DATABASE_* → DB_* fallback): `CACHE_DB_URL` if set; else, when `get_relational_config().db_provider == "postgres"`, build `postgresql+asyncpg://DB_USERNAME:DB_PASSWORD@DB_HOST:DB_PORT/DB_NAME` with a warning-level log; else raise `CacheConnectionError("CACHE_BACKEND=postgres requires CACHE_DB_URL or DB_PROVIDER=postgres")`.
- Extend `to_dict()` and the class docstring (both pinned by `cognee/tests/unit/infrastructure/databases/cache/test_cache_config.py::test_cache_config_defaults/test_cache_config_to_dict`, which assert the exact full dict — update them).

**`cognee/infrastructure/databases/cache/get_cache_engine.py`:**
- Thread two new hashable params through both signatures (same pattern as the `tapes_*` params): `create_cache_engine(..., cache_db_url: str | None = None, cache_purge_interval_seconds: int = 900)`; pass from `get_cache_engine()`.
- New branch with **lazy import** (verified pattern: `TapesCacheAdapter` is imported inside its branch):

```python
elif config.cache_backend == "postgres":
    from cognee.infrastructure.databases.cache.postgres.PostgresCacheAdapter import (
        PostgresCacheAdapter,
    )
    return PostgresCacheAdapter(
        connection_string=_resolve_cache_db_url(cache_db_url),
        lock_key=lock_key,
        log_key=log_key,
        session_ttl_seconds=session_ttl_seconds,
        agentic_lock_expire=agentic_lock_expire,
        agentic_lock_timeout=agentic_lock_timeout,
        purge_interval_seconds=cache_purge_interval_seconds,
    )
```

- Update the `ValueError` message to `"'redis', 'fs', 'tapes', 'postgres'"`.
- Move the currently-unconditional `from ...RedisAdapter import RedisAdapter` (verified: executes for ALL backends once caching is on) inside the `== "redis"` branch — 2-line cleanup that removes the accidental reliance on `fakeredis` pulling in `redis`. Low-risk, do it in this PR.
- lru_cache caveats respected: `cache_db_url` is a hashable string; constructor params frozen after first call until `cache_clear()` (existing behavior); note the existing pool-per-`lock_key` instantiation pattern (Ladybug per-db lock_key) in the adapter docstring.

**`pyproject.toml`:** no changes — SQLAlchemy is core; asyncpg/psycopg2 are in the existing `postgres` extra. Document "`CACHE_BACKEND=postgres` requires `cognee[postgres]`".

---

## 6. Schema creation / migration approach

Graph-adapter style, **not alembic**:
- Tables on a private `MetaData()` in `cache/postgres/tables.py`; alembic's `env.py` (`target_metadata = Base.metadata`) never sees them; no migration files.
- Idempotent `create_all(checkfirst=True)` inside `_ensure_initialized()` (once-flag + `asyncio.Lock`), re-run lazily after `close_cache_engine()`/`prune_system`'s `cache_clear()` recreates the adapter — necessary because the cache factory has no `_GraphEngineHandle` equivalent.
- Rationale: cache content is ephemeral/TTL'd, schema is additive, and the repo precedent is explicit (`graph_node`/`graph_edge` create-on-init vs alembic-managed `session_records`).
- **No per-dataset database handler** — do not register anything in `supported_dataset_database_handlers.py` or copy `PostgresGraphDatasetDatabaseHandler`. Sessions are per-user, not per-dataset; isolation parity comes from `(user_id, session_id)` columns. Deployments wanting physical separation point `CACHE_DB_URL` at a dedicated database.

---

## 7. Session lock handling

- **`cognee/infrastructure/locks/session_lock.py`: no change.** It is pure in-process asyncio with no Redis dependency; its documented single-worker scope is unchanged. Note in the PR description that the adapter's `FOR UPDATE` in `update_qa_entry`/`delete_feedback` makes those RMW paths multi-worker-safe at the storage layer anyway (closes the gap session_lock.py's own docstring flags).
- **`CacheDBInterface.acquire_lock`/`release_lock` (shared Ladybug lock):**
  - **v1 (core PR):** raise `SharedLadybugLockRequiresRedisError` — established FS precedent; `SHARED_LADYBUG_LOCK=true + CACHE_BACKEND=postgres` stays unsupported with a clear error.
  - **Phase 6 (separate PR):** lazy-create a small **sync** engine (`postgresql+psycopg2`, `pool_size=2`; psycopg2 is already in the `postgres` extra) used only for locks. `lock_id = int.from_bytes(sha256(self.lock_key.encode()).digest()[:8], "big", signed=True)`. `acquire_lock()`: check out a dedicated connection, loop `SELECT pg_try_advisory_lock(:id)` with 0.1 s sleeps until the `agentic_lock_timeout` deadline; on success return a handle object owning the connection; on timeout raise `RuntimeError` (mirrors Redis). `release_lock(lock=None)`: `pg_advisory_unlock` on the **passed** handle, not `self.lock` (explicitly tested Redis behavior); close/return the connection; swallow errors if not held. Handle-bound connections give `thread_local=False` parity (releasable from worker threads). Semantics note to document: advisory locks release on connection death (safer than, but different from, Redis's 240 s `agentic_lock_expire` auto-expiry, which has no direct equivalent).
- Multi-worker `improve()` mutex via `pg_try_advisory_lock` behind `try_acquire_improve_lock`: explicit non-goal / future work.

---

## 8. Test plan

**Unit CRUD** — new `cognee/tests/unit/infrastructure/databases/cache/test_postgres_adapter_crud.py`, mirroring `test_redis_adapter_crud.py`'s ~25 cases. No real Postgres in pytest (repo convention — Redis suites use a hand-rolled `_InMemoryRedisList` fake; here we get a real-engine equivalent for free): construct `PostgresCacheAdapter("sqlite+aiosqlite:///<tmp>/cache.db")` — the `with_variant(JSON(), "sqlite")` columns and identity-PK→autoincrement degradation make the same code paths run without a server. Cover:
- QA create/get-latest/get-all ordering; uuid4 `qa_id` fallback; `[]` on empty **including `last_n=1`** (pins the FS-style choice)
- update: None-preserves every field; `memify_metadata` MERGE; `SessionQAEntryValidationError` propagates unwrapped; `False` on missing qa_id
- `delete_feedback` nulls both fields; `delete_qa_entry` rowcount semantics + TTL survival; `delete_session` clears both tables
- traces: append/get/count/feedback delegation; sanitization via model validators; blank `trace_id` → `CacheConnectionError`
- TTL: `expires_at` set/refreshed on create/update/delete_feedback/after-delete; NOT refreshed on reads; disabled when `session_ttl_seconds in (0, None)`; expired rows invisible to reads (assert via direct SQL with backdated `expires_at`)
- `prune()` clears only the four cache tables; `close()` idempotent; legacy `add_qa`/`get_latest_qa`/`get_all_qas` shims
- `acquire_lock`/`release_lock` raise `SharedLadybugLockRequiresRedisError`
- KV `get_value`/`set_value`/`delete_value` round-trips, including one legacy `graph_knowledge:{u}:{s}` deletion path through `SessionManager.delete_session`

**Integration** — add `"postgres"` to the `params=["fs", "redis"]` fixtures in `cognee/tests/integration/infrastructure/session/test_session_sdk_integration.py`, `test_session_persistence_memify_integration.py`, `test_feedback_weights_memify_integration.py` (aiosqlite URL, no server); new `test_session_manager_postgres.py` mirroring `test_session_manager_redis.py` (LLM mocked via `AsyncMock` on `LLMGateway.acreate_structured_output`).

**Config/factory** — update `test_cache_config.py` exact-dict assertions; new factory tests: `CACHE_BACKEND=postgres` returns `PostgresCacheAdapter`, unknown backend `ValueError` lists all four, URL fallback resolution (CACHE_DB_URL → DB_* → error).

**CI e2e (real Postgres)** — third twin job `run_conversation_sessions_test_postgres` in `.github/workflows/e2e_tests.yml`, copying `run_conversation_sessions_test_redis` minus the redis service (the pgvector/pg17 service is already provisioned there): env `CACHING=true AUTO_FEEDBACK=true CACHE_BACKEND=postgres DB_PROVIDER=postgres` (relying on the DB_* fallback, which also exercises it) or explicit `CACHE_DB_URL`; extras `"postgres"` only; runs `cognee/tests/test_conversation_history.py` unmodified. Phase 6 adds a `SHARED_LADYBUG_LOCK=true CACHE_BACKEND=postgres` variant of the concurrent-subprocess job running `test_concurrent_subprocess_access.py`, plus real-Postgres-gated advisory-lock unit tests (acquire/release of the *passed* handle, contention timeout → `RuntimeError`, cross-connection mutual exclusion).

**Regression coverage** — FS/Redis/Postgres tests proving generic string KV round-trips and the legacy `graph_knowledge:` cleanup path work consistently.

---

## 9. Rollout & backward compatibility

- **No behavior change by default:** `cache_backend` default stays `"fs"`; redis/tapes branches untouched; `get_cache_engine()` still returns `None` when caching + usage_logging are off (every consumer already handles `None`).
- **No data migration tooling:** session cache is 7-day-TTL ephemeral; switching backends starts fresh — same story as redis↔fs today. One-liner in docs. `SessionRecord` rows in the relational DB are unaffected.
- **`forget(everything=True)` / `prune_system` / `prune_data`** paths work unchanged (`prune()`, `close_cache_engine()`, `cache_clear()`); document that Postgres `prune()` is scoped to cognee cache tables (an improvement over `FLUSHDB`).
- **Failure modes:** misconfigured backend raises `CacheConnectionError` (503) at first use; `SHARED_LADYBUG_LOCK=true` + postgres raises `SharedLadybugLockRequiresRedisError` with a clear message until Phase 6.
- **Docs:** `.env.template` "Session cache settings" block (~line 375): add `# CACHE_BACKEND=postgres` + `# CACHE_DB_URL=postgresql+asyncpg://...`; `config.py` docstring backend list; factory `ValueError` message; `CLAUDE.md` extras note ("postgres — also enables the Postgres session-cache backend"); file a task for the external docs.cognee.ai "sessions-and-caching" page. `docker-compose.yml` needs nothing (postgres service exists; redis stays behind its profile).

---

## 10. Phased work breakdown

| Phase | Scope | Size |
|---|---|---|
| **P1 — Adapter core** | `cache/postgres/tables.py` + `PostgresCacheAdapter.py`: engine, `_ensure_initialized`, QA CRUD (`FOR UPDATE` updates, atomic deletes), TTL refresh + read filter + scoped/throttled purge, prune, close | **L** (~1.5–2 days, ~450 LOC) |
| **P2 — Traces, usage logs, locks-raise, string KV** | `append_agent_trace_step`/trace reads, `log_usage`/`get_usage_logs` (whole-list TTL refresh), lock methods raising `SharedLadybugLockRequiresRedisError`, `get_value`/`set_value`/`delete_value` | **M** (~0.5–1 day, ~150 LOC) |
| **P3 — Wiring** | `config.py` Literal + `cache_db_url` + `cache_purge_interval_seconds` + `to_dict` + docstring; `get_cache_engine.py` branch + `_resolve_cache_db_url` fallback + error msg + RedisAdapter import moved into its branch; config-test fixes | **S** (~0.5 day) |
| **P4 — Tests** | Unit CRUD suite (aiosqlite), `test_session_manager_postgres.py`, `"postgres"` params in 3 integration files, factory/config tests | **M** (~1 day, ~500 LOC tests) |
| **P5 — CI + docs** | `run_conversation_sessions_test_postgres` e2e twin job, `.env.template`, `CLAUDE.md`, docs.cognee.ai task | **S** (~0.5 day) |
| **P6 — Follow-up PR: advisory locks** | psycopg2 `pg_try_advisory_lock` sync lock impl lifting the `SHARED_LADYBUG_LOCK` restriction + concurrent-subprocess CI variant | **M** (~1 day) |

Core (P1–P5): **~4 days**, one reviewable PR (or two: adapter+tests, then wiring+CI). P6 ships separately.

---

## 11. Future: Turbopuffer adapter sketch (do not build now — verify the seam)

Turbopuffer is a namespaced object/vector store with upsert-by-id, delete-by-id, and attribute-filtered queries. The same `CacheDBInterface` maps because every cognee cache access is by exact composite key, append-or-point-update, with no cross-key scans:

- **Namespaces:** today's key strings verbatim (`agent_sessions:{user_id}:{session_id}`, `agent_traces:{...}`, `usage_logs:{user_id}`) — key-prefix tenancy carries over unchanged.
- **Rows:** id = `qa_id` / trace uuid; attributes = entry fields plus a client-stamped monotonic `seq`/`created_at` for ordering (no serial column); vector optional (zero/1-dim placeholder if mandatory — or embed `question+answer` for free semantic recall later, a genuine upside).
- **Operations:** create/append → upsert; `get_latest/get_all` → seq-ordered query (tail = desc + limit + reverse); `update_qa_entry`/`delete_feedback` → read-merge-upsert by id (back to Redis-grade RMW races — no `FOR UPDATE`; acceptable for a cache tier, flag in docs); `delete_session` → namespace delete; `prune()` → enumerate + delete `cognee-*` namespaces by prefix; KV → a `cache_kv` namespace with key-as-id rows; TTL → `expires_at` attribute + query filter + periodic GC (no native TTL — the Postgres TTL design transfers directly); locks → raise `SharedLadybugLockRequiresRedisError` (precedent).
- **Seam requirements this plan satisfies:** (1) all storage details stay behind `CacheDBInterface`; no caller touches adapter internals; (2) entries cross the boundary only as pydantic `model_dump()` payloads; (3) ordering is adapter-internal (`seq` column vs `seq` attribute) — nothing leaks; do not let SQL row ids escape into return values during P2; (4) wiring is mechanical: `cache_backend` Literal += `"turbopuffer"`, lazy-import elif, `TURBOPUFFER_API_KEY`/`TURBOPUFFER_REGION` threaded through the lru_cache'd factory.

---

## 12. Open questions

1. **Default fallback to the relational DB:** when `CACHE_DB_URL` is unset and `DB_PROVIDER=postgres`, the cache silently shares the relational database (distinct `cache_*` tables, warning logged). Is co-tenancy acceptable as the default, or should `CACHE_DB_URL` be mandatory?
2. **Normalize the Redis `None`-on-`last_n==1` quirk in `RedisAdapter` itself** (return `[]` like FS/Postgres), or leave Redis as-is and only document Postgres's `[]` choice? (Some tests pin the Redis behavior.)
3. **Hot-path payload size:** should `get_latest_qa_entries` eventually project `payload - 'context'` for history reads (formatted history uses `include_context=False` but still fetches full entries)? Behavior-preserving today; revisit with real latency data.
4. **Throttled global purge tuning:** is 900 s / per-process advisory-lock-guarded sweep enough for high-volume deployments, or is a documented external cron (`DELETE ... WHERE expires_at <= now()`) preferable at scale?
5. **`fakeredis[lua]` core dependency:** verified unused in-repo and only load-bearing because it transitively installs `redis` for the unconditional import this plan removes — candidate for deletion in a separate cleanup PR (needs a check that no downstream consumers rely on it).
6. **Lock auto-expiry parity (Phase 6):** Redis locks auto-expire after `agentic_lock_expire=240` s; pg advisory locks hold until connection death. Is connection-scoped release acceptable, or do we need a watchdog that closes the lock connection after 240 s?
