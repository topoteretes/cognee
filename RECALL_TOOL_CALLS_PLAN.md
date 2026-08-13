# Tool Calls in cognee.recall — Implementation Plan (text-to-SQL first)

## 1. Decision summary

Three design lenses (minimal / extensible / security) were evaluated against the actual code. The synthesis:

**Adopt the minimal core, not a new `RecallTool` protocol.** The codebase already has a tool subsystem — `cognee/modules/tools/` contains `execute_tool()` (registry lookup + per-dataset ACL via `get_authorized_existing_datasets`), `registry.py` (`register_builtin_tool`, `_BUILTIN_TOOLS`), the `Tool` DataPoint (`cognee/modules/engine/models/Tool.py`, whose docstring explicitly anticipates a SQL toolset), and a working ReAct loop in `AgenticRetriever._run_tool_loop`. Introducing a second, recall-specific `RecallTool` registry (the extensible proposal) would create two parallel tool registries with acknowledged drift risk. Instead:

- recall gets a **`"tools"` scope + one `_run_tools` runner** in the existing runner-dict architecture (verified: `runners = {...}` at `cognee/api/v1/recall/recall.py:600`),
- new tool code lives **inside the existing `cognee/modules/tools/` package** (`text_to_sql/` sub-package, `config.py`, `models/`),
- extensibility comes from a **generic `ResponseToolEntry`** (`tool_name` discriminates tools, `structured` dict carries tool-specific payload) and from a later phase that registers `text_to_sql` as a builtin `Tool`, making it callable from the agentic path with dataset-ACL enforcement for free.

**Graft all of the security lens's non-negotiables** (we execute LLM-generated SQL against customer databases):

1. Master gate `TOOL_CALLS_ENABLED=false` by default (BaseSettings, not the raw-`os.getenv` `ALLOW_CYPHER_QUERY` default-true pattern at `get_search_type_retriever_instance.py:374`).
2. Per-user **authorized connection registry** with DSNs encrypted at rest via the existing AES-256-GCM keyring (`cognee/modules/integrations/crypto.py`, verified `encrypt_credentials`/`decrypt_credentials` and the `ciphertext/nonce/key_id` column pattern in `IntegrationCredential.py`).
3. Three-layer **read-only enforcement**: SELECT-only SQL guard → read-only connection args + a rollback-only executor (never `SqlAlchemyAdapter.execute_query`, verified at line 559/573 to run inside `engine.begin()`, which commits) → documented SELECT-only DB role.
4. **Audit**: every executed SQL recorded in observability spans and, when `session_id` is present, in the session trace.
5. `"tools"` is **never** part of `scope="all"` or `scope="auto"` — no existing caller silently reaches an external database, and older cloud servers (which forward the *resolved* list for `"all"`, verified `_scope_should_forward_resolved` at `recall.py:330-333`) never receive an unknown source.

**Phasing**: 4 independently-shippable PRs; PR 1 is the smallest end-to-end vertical slice (register one authorized DB via SDK → `recall(scope=["tools"])` → tagged `RecallResponse` entries come back).

---

## 2. Verified integration points (all paths checked in-repo)

| Concern | File / symbol | Status |
|---|---|---|
| Scope validation | `cognee/memory/entries.py` — `RecallScope` (l.138), `_VALID_SCOPES` (l.144), `normalize_scope` | exists |
| Runner dispatch | `cognee/api/v1/recall/recall.py` — `recall()` (l.336), auto-expansion (l.409-423), remote forwarding (l.451-474), `runners` dict (l.600), `auto_fallthrough` short-circuit (l.613) | exists |
| Response union | `cognee/modules/recall/types/RecallResponse.py` — `Annotated[... , Field(discriminator="source")]` over 4 entries | exists |
| HTTP endpoint | `cognee/api/v1/recall/routers/get_recall_router.py` — `RecallPayloadDTO` (l.23), free-form `scope` field (l.78), `response_model=list[RecallResponse]` | exists |
| Cloud client | `cognee/api/v1/serve/cloud_client.py` — `CloudClient.recall` (l.189), explicit payload whitelist (l.195-217, `scope` already forwarded at l.213) | exists |
| CLI client | `cognee/cli/api_client.py` — `CogneeApiClient.recall` (l.312), **no** `scope` param today | exists |
| NL→query→retry template | `cognee/modules/retrieval/natural_language_retriever.py` — `_generate_cypher_query` (l.53), `max_attempts` loop (l.75+), `previous_attempts` feedback | exists |
| Prompt template | `cognee/infrastructure/llm/prompts/natural_language_retriever_system.txt` | exists |
| External-DB engine | `cognee/infrastructure/databases/relational/create_relational_engine.py` — `@lru_cache` factory (l.9-10) with `database_connect_args` plumbing (l.18, 94, 105); `get_migration_relational_engine.py` | exists |
| SQL primitives | `cognee/infrastructure/databases/relational/sqlalchemy/SqlAlchemyAdapter.py` — `execute_query` (l.559, **commits** via `engine.begin()`), `extract_schema` (~l.692) | exists |
| Secrets crypto | `cognee/modules/integrations/crypto.py` — `encrypt_credentials` (l.89), `decrypt_credentials` (l.103), `INTEGRATION_CREDENTIALS_KEYS` keyring | exists |
| Tool subsystem | `cognee/modules/tools/` — `execute_tool.py`, `registry.py`, `errors.py`, `builtin/memory_search.py`; `cognee/modules/engine/models/Tool.py` | exists |
| Tests | `cognee/tests/unit/api/v1/recall/test_recall_api.py`, `cognee/tests/unit/modules/tools/` | exist |

---

## 3. API surface

### 3.1 Registering an authorized database

**Python SDK** (new `cognee/api/v1/tools/tools.py`, exported as `cognee.tools`):

```python
await cognee.tools.register_sql_connection(
    name="analytics",                       # unique per owner
    connection_string="postgresql+asyncpg://ro_user:pw@host:5432/analytics",
    provider=None,                          # inferred; "postgres" | "sqlite" in v1
    allowed_tables=["orders", "customers"],  # optional schema/prompt scope cap
    max_rows=100,                            # optional per-connection override
    description="Analytics warehouse (read-only role)",
    user=None,                               # None → default user in single-user mode
)
await cognee.tools.list_sql_connections(user=None)    # never returns the DSN
await cognee.tools.remove_sql_connection("analytics", user=None)
```

DSN is AES-256-GCM-encrypted at rest via `cognee/modules/integrations/crypto.py`; registration fails with a clear error if `INTEGRATION_CREDENTIALS_KEYS` is unset (no plaintext fallback — deliberately not the `DatasetDatabase` plaintext-column or `UserApiKey` optional-hash posture).

**Deployment-level env alternative** (single-tenant convenience; reachable by *every* authenticated recall caller — documented caveat):

```bash
TOOL_CALLS_ENABLED=true                     # master gate, default false
TOOL_SQL_CONNECTIONS='{"analytics": {"provider": "postgres", "host": "db", "port": 5432, "username": "ro_user", "password": "...", "name": "analytics_db"}}'
TOOL_TEXT_TO_SQL_MAX_ROWS=100
TOOL_TEXT_TO_SQL_MAX_ATTEMPTS=3
TOOL_TEXT_TO_SQL_STATEMENT_TIMEOUT_MS=5000
TOOL_TEXT_TO_SQL_MAX_SCHEMA_TABLES=50
```

**HTTP** (PR 2, new router on the forget/integrations template, behind `get_authenticated_user`):

```
POST   /api/v1/tools/connections     {"name", "connectionString", "allowedTables", "maxRows", "description"}
GET    /api/v1/tools/connections     → non-secret listing
DELETE /api/v1/tools/connections/{name}
```

### 3.2 Invoking through recall

**SDK** — `recall()` gains one keyword param `tool_connections: list[str] | None = None`:

```python
results = await cognee.recall(
    "What was total revenue by region last quarter?",
    scope=["tools"],                  # or ["graph", "tools"]; never implied by None/"auto"/"all"
    tool_connections=["analytics"],   # None → all connections visible to this user
    session_id="s1",                  # optional: audits executed SQL into the session trace
)
```

**HTTP** — `POST /api/v1/recall` with `{"query": "...", "scope": ["tools"], "toolConnections": ["analytics"]}` (`RecallPayloadDTO` gains `toolConnections`; `scope` already passes arbitrary strings through — verified l.78).

**CLI** (PR 3) — `cognee-cli recall "question" --scope tools --tool-connections analytics` (both `cognee/cli/api_client.py` `recall()` — which today has *no* scope param — and `recall_command.py` need additions).

### 3.3 Response model

One new member of the `RecallResponse` discriminated union (`cognee/modules/recall/types/RecallResponse.py`); the router's `response_model=list[RecallResponse]` picks it up transitively:

```python
class ResponseToolEntry(BaseModel):
    source: Literal["tools"]
    tool_name: str                    # "text_to_sql" in v1; discriminates future tools
    question: str
    text: str                         # rendered compact answer / row table (row-capped, never mid-row truncated)
    success: bool = True
    error: Optional[str] = None       # per-connection execution failure; authz failures raise instead
    structured: Optional[dict] = None # text_to_sql: {"connection", "dialect", "sql",
                                      #   "rows": [...JSON-safe...], "row_count", "truncated", "attempts"}
```

Generic by design: tool #2 (HTTP API, MCP, …) reuses this entry with its own `structured` payload — no union change, no `entries.py` change, no runner change.

---

## 4. Architecture — new and modified code

### New: `cognee/modules/tools/config.py`
`ToolsConfig(BaseSettings)` on the `CognifyConfig` house pattern (`SettingsConfigDict(env_file=".env", extra="allow")`, `@lru_cache get_tools_config()`, `to_dict()`): `tool_calls_enabled: bool = False`, `tool_sql_connections: str = "{}"` (JSON env, parsed like `RelationalConfig`'s `DATABASE_CONNECT_ARGS`), `text_to_sql_max_rows=100`, `text_to_sql_max_attempts=3`, `text_to_sql_statement_timeout_ms=5000`, `text_to_sql_max_schema_tables=50`.

### New: `cognee/modules/tools/models/ToolConnection.py`
SQLAlchemy model on `Base` (`cognee/infrastructure/databases/relational/ModelBase.py`), auto-created by `Base.metadata.create_all` in `SqlAlchemyAdapter.create_database` once imported (mirror how `cognee/modules/users/models/__init__.py` exports models; no alembic exists in-repo). Columns: `id`, `user_id` (indexed owner), `name` (UNIQUE(user_id, name)), `provider`, `ciphertext/nonce/encryption_version/key_id` (encrypted DSN payload — column set copied from `IntegrationCredential`), `options` JSON (non-secret: `allowed_tables`, `max_rows`, `description`), `status`, timestamps.

A dedicated table is preferred over overloading `IntegrationCredential` rows (`provider="external_sql"`): clean per-user UNIQUE name, no OAuth semantics, no `provider_account_id=f"{owner_id}:{name}"` workaround for the global UNIQUE(provider, provider_account_id) index. The crypto module is shared either way. (Flagged as an open question for maintainers.)

### New: `cognee/modules/tools/connections.py`
CRUD modeled on `cognee/modules/integrations/credentials.py`: `register_tool_connection` / `list_tool_connections` / `get_tool_connection(user_id, name)` / `delete_tool_connection`. `get_tool_connection` **fails closed** with `ToolPermissionError` (existing taxonomy in `cognee/modules/tools/errors.py`) on non-owned names — explicitly not recall's swallow-exceptions session-owner pattern. Also merges deployment-level env connections (owner = deployment; visible to all callers). Provider inference reuses `DB_CONNECTION_PATTERNS`/`is_connection_string` from `cognee/tasks/ingestion/create_dlt_source.py`.

### New: `cognee/modules/tools/text_to_sql/` package
- **`sql_guard.py`** — `validate_select(sql, dialect)`: comment/fence stripping, exactly one statement (reject embedded `;`), first token ∈ {SELECT, WITH}, word-boundary denylist outside string literals (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/GRANT/REVOKE/COPY/ATTACH/VACUUM/PRAGMA/CALL/MERGE/EXECUTE, incl. CTE bodies); `ensure_limit(sql, max_rows)` injects/clamps LIMIT. `sqlglot` parse when installed, strict token fallback otherwise (dependency decision is an open question). Raises `SqlGuardError`. Pure functions, heavily unit-tested.
- **`engine_factory.py`** — builds the external engine from the decrypted DSN via the existing `@lru_cache`'d `create_relational_engine()` (pooled per-connection engines for free, the `get_migration_relational_engine.py` pattern), passing read-only connect args through the verified `database_connect_args` plumbing: postgres/asyncpg `server_settings={"default_transaction_read_only": "on", "statement_timeout": ...}`; sqlite URI `mode=ro`.
- **`executor.py`** — `execute_readonly(adapter, sql, max_rows, timeout_ms)`: opens `engine.connect()` (never the committing `engine.begin()` that `SqlAlchemyAdapter.execute_query` uses), issues `SET TRANSACTION READ ONLY` on postgres, executes `text(sql)`, fetches at most `max_rows + 1` (sets `truncated`), **always rolls back**, and coerces rows JSON-safe (Decimal/datetime/date/UUID/bytes).
- **`engine.py`** — `run_text_to_sql(user, connection_name, question)`: the `NaturalLanguageRetriever` loop ported to SQL — gate check → `get_tool_connection` (authz) → schema via `SqlAlchemyAdapter.extract_schema()` filtered by `allowed_tables` and capped at `text_to_sql_max_schema_tables` → `render_prompt("text_to_sql_system.txt", {schema, dialect, max_rows, previous_attempts})` → `LLMGateway.acreate_structured_output(response_model=GeneratedSql)` → guard → `execute_readonly` → retry feeding guard/execution errors back as `previous_attempts` up to `max_attempts`. Two deliberate deviations from the Cypher loop: an **empty result set is success** (no wasted LLM retries), and exhausted attempts return an explicit error result (not a silent `[]`).
- **New prompt** `cognee/infrastructure/llm/prompts/text_to_sql_system.txt` (Jinja2, on the `natural_language_retriever_system.txt` model): schema-grounded, "return ONLY one SELECT", dialect note, LIMIT guidance, previous-attempts analysis, treat-schema-names-as-data instruction. No sample rows in the prompt by default (exfiltration / prompt-injection surface).

### Modified: recall wiring
- `cognee/memory/entries.py` — add `"tools"` to `_VALID_SCOPES` (l.144) and the `RecallScope` Literal (l.138). Do **not** add to the `"all"` expansion, do **not** touch the auto-expansion block (`recall.py:409-423`).
- `cognee/modules/recall/types/RecallResponse.py` — append `ResponseToolEntry` to the union.
- `cognee/api/v1/recall/recall.py` — `tool_connections` kwarg on `recall()`; `_run_tools` closure registered as `runners["tools"]` (l.600): gate check raises clearly when the scope was explicitly requested with the feature off (avoiding the `runners.get(src) → continue` silent-`[]` trap), resolves the user, runs `run_text_to_sql` per connection with per-connection try/except (DB/LLM failures → `success=False` entries so a dead external DB never aborts a `["graph","tools"]` recall; authorization failures raise), sets span attributes, appends a session-trace audit entry when `session_id` is set. Forward `tool_connections` in the remote-client call (l.457-471).
- `cognee/api/v1/recall/routers/get_recall_router.py` — `toolConnections` field on `RecallPayloadDTO`, threaded through; scope docs updated; convert `normalize_scope`'s `ValueError` into a 422 with the valid-scope list (today it becomes an opaque 409 via the generic handler).
- `cognee/api/v1/serve/cloud_client.py` — add `tool_connections` to the explicit payload whitelist (l.195-217; `scope` already forwards).

### Later (PR 4): agentic bridge
`cognee/modules/tools/builtin/text_to_sql.py` mirroring `memory_search.py`: module-level `Tool(name="text_to_sql", input_schema={question, connection, max_rows}, handler_ref=..., permission_required="read", readonly_hint=True)` + `register_builtin_tool` at import. Handler **re-checks** `TOOL_CALLS_ENABLED` and connection ownership itself — defense against the factory-only-gate bypass that `ALLOW_CYPHER_QUERY` suffers. Makes text-to-SQL reachable via `search(query_type=SearchType.AGENTIC_COMPLETION, retriever_specific_config={"tools": ["text_to_sql"]})` with `execute_tool`'s dataset ACL on top. One executor (`run_text_to_sql`), two entry points.

---

## 5. Text-to-SQL flow (end to end)

1. `recall(query, scope=["tools"], tool_connections=[...])` → `normalize_scope` validates → `_run_tools` runs in the existing sequential merge loop.
2. **Gate**: `get_tools_config().tool_calls_enabled` must be true → explicit-scope requests fail loudly otherwise.
3. **AuthZ**: resolve calling user (default user when `ENABLE_BACKEND_ACCESS_CONTROL=false` — documented, deliberate); each requested name resolved via `get_tool_connection(user_id, name)`; non-owned name → `ToolPermissionError` (fail closed). Default = all connections visible to the user (own rows ∪ deployment env connections).
4. **Connect**: `decrypt_credentials(...)` → DSN → `create_relational_engine(...)` with read-only connect args; lru_cache keeps the pooled engine warm.
5. **Schema**: `extract_schema()` → `{table: {columns, primary_key, foreign_keys}}`, filtered by `allowed_tables`, capped at `text_to_sql_max_schema_tables`.
6. **Generate**: `render_prompt` + `LLMGateway.acreate_structured_output(response_model=GeneratedSql)`.
7. **Guard**: `validate_select` + `ensure_limit`; rejection counts as an attempt and is fed back to the LLM.
8. **Execute**: `execute_readonly` — read-only transaction, fetch `max_rows+1`, unconditional rollback, JSON-safe coercion. Errors → `previous_attempts` → retry from step 6, bounded by `max_attempts`. Empty result = success.
9. **Audit**: span attributes (`cognee.recall.tool`, connection name, SQL hash, row_count, duration, outcome) via `new_span`; session-trace entry (question + generated SQL + row_count; never the DSN) when session-scoped.
10. **Shape & merge**: `ResponseToolEntry(source="tools", tool_name="text_to_sql", text=<compact table>, structured={connection, dialect, sql, rows, row_count, truncated, attempts})` merged with other sources and serialized through the extended union — SDK and `POST /api/v1/recall` identically.

---

## 6. Security model

| Layer | Mechanism |
|---|---|
| Feature gate | `TOOL_CALLS_ENABLED=false` default (BaseSettings). Checked in `_run_tools` **and** inside `run_text_to_sql` / the builtin handler (no bypass via the agentic path). |
| Authorization | Per-user-owned `ToolConnection` rows (the `IntegrationCredential` ownership contract); `get_tool_connection` fails closed. Env-level connections are deployment-scoped trust (same as `MigrationConfig`) — documented as unsuitable for multi-tenant. Cross-user sharing deferred: `ACL.dataset_id` is a hard FK to `datasets.id`, so a future parallel `ToolACL` table mirroring `give_permission_on_dataset` / `get_specific_user_permission_datasets` is the designed seam. Agentic path additionally inherits dataset ACL via `execute_tool`. |
| Read-only (3 layers) | (1) static SELECT-only guard; (2) connection-level `default_transaction_read_only=on` / sqlite `mode=ro` + `SET TRANSACTION READ ONLY` + rollback-only executor (never `execute_query`, which commits); (3) docs + `.env.template` mandate a SELECT-only DB role. |
| Resource limits | injected/clamped LIMIT, `max_rows+1` fetch cap with `truncated` flag (row caps, never mid-row string truncation), per-statement timeout, schema-table cap in the prompt, bounded retry loop. |
| Secrets | DSNs encrypted at rest (AES-256-GCM keyring, key rotation via `key_id`); decrypted only at point of use; never in Tool DataPoints (descriptions are embedded into vector stores), API responses, spans, logs, traces, or `ResponseToolEntry`. No plaintext fallback. |
| Prompt injection | Schema names / error text flow into the prompt with treat-as-data instructions, but enforcement never relies on the prompt: guard + read-only transaction apply to every statement. No sample rows in prompts; query results are not fed back into any LLM call in v1. |
| Audit | Every execution → span attributes + session-trace entry; recall's trace scope makes the audit itself recallable. |
| Scope safety | `"tools"` excluded from `"all"`/`"auto"`; explicit opt-in per call. Also avoids older cloud servers 409-ing on a resolved `"all"` list containing an unknown source. |

---

## 7. Phased delivery (independently-shippable PRs)

Branch from `dev`; every PR carries a Linear key (`COG-xxx`) — the current branch `Vasilije1990/add_tool_call_ability` has none and will fail the required `linear-issue-check`. Run `pre-commit run --all-files` before each commit.

### PR 1 — Vertical slice: authorized connections + text-to-SQL + recall `"tools"` scope
Smallest end-to-end: `cognee.tools.register_sql_connection(...)` → `recall(query, scope=["tools"])` → `ResponseToolEntry` results, SDK and HTTP.

Create:
- `cognee/modules/tools/config.py` (ToolsConfig, gate default-off)
- `cognee/modules/tools/models/__init__.py`, `cognee/modules/tools/models/ToolConnection.py`
- `cognee/modules/tools/connections.py` (CRUD, fail-closed authz, env-connection merge)
- `cognee/modules/tools/text_to_sql/__init__.py`, `sql_guard.py`, `engine_factory.py`, `executor.py`, `engine.py`
- `cognee/infrastructure/llm/prompts/text_to_sql_system.txt`
- `cognee/api/v1/tools/__init__.py`, `cognee/api/v1/tools/tools.py` (SDK registration functions; export as `cognee.tools` from `cognee/__init__.py`)

Modify:
- `cognee/memory/entries.py` (`"tools"` in `_VALID_SCOPES` + `RecallScope`; not in `"all"`)
- `cognee/modules/recall/types/RecallResponse.py` (`ResponseToolEntry`)
- `cognee/api/v1/recall/recall.py` (`tool_connections` kwarg, `_run_tools`, runner registration, remote-forwarding)
- `cognee/api/v1/recall/routers/get_recall_router.py` (`toolConnections`, scope docs, ValueError→422)
- `cognee/api/v1/serve/cloud_client.py` (whitelist `tool_connections`)
- `cognee/modules/tools/errors.py` (`SqlGuardError`, `ToolConnectionNotFoundError`)
- `.env.template` (TOOL_CALLS_ENABLED + TOOL_TEXT_TO_SQL_* + INTEGRATION_CREDENTIALS_KEYS note)

Tests:
- `cognee/tests/unit/modules/tools/test_tools_config.py` (defaults, lru_cache-clearing env fixtures)
- `cognee/tests/unit/modules/tools/test_sql_guard.py` (DML/DDL/multi-statement/CTE-INSERT/comment-obfuscation rejection; LIMIT injection/clamping)
- `cognee/tests/unit/modules/tools/test_tool_connections.py` (ownership isolation, encryption round-trip, missing-keyring error, no secrets in listings)
- `cognee/tests/unit/modules/tools/test_readonly_executor.py` (rollback, truncation flag, JSON coercion, write blocked at DB level on sqlite `mode=ro`)
- `cognee/tests/unit/modules/tools/test_text_to_sql_engine.py` (mocked `LLMGateway` + temp sqlite: happy path, retry-on-error, guard-rejection feedback, empty-result-no-retry, exhausted-attempts error entry)
- extend `cognee/tests/unit/api/v1/recall/test_recall_api.py` (scope validation, gate-off raises, runner registered — no silent skip, union serialization round-trip, router 422, cloud kwarg forwarding, one failing connection doesn't abort graph results)

### PR 2 — HTTP registration API
Create: `cognee/api/v1/tools/routers/__init__.py`, `cognee/api/v1/tools/routers/get_tools_router.py` (forget/integrations router shape: `InDTO` camelCase payloads, `@log_usage`, `send_telemetry`, `get_authenticated_user`, lazy imports, error→JSONResponse). Modify: `cognee/api/client.py` (`include_router(..., prefix="/api/v1/tools", tags=["tools"])`). Tests: `cognee/tests/unit/api/v1/tools/test_tools_router.py` (auth required, secret-masking assertions, CRUD wiring).

### PR 3 — CLI + docs + integration test
Modify: `cognee/cli/api_client.py` (`recall()` gains `scope` + `tool_connections` — it has neither today), `cognee/cli/commands/recall_command.py` (`--scope`, `--tool-connections`, `source=="tools"` rendering branch: connection, SQL, row table, error), `CLAUDE.md` (Security Considerations + recall scope docs), `.env.template` polish. Tests: `cognee/tests/cli_tests/test_recall_command_tools.py`; `cognee/tests/integration/tools/test_recall_text_to_sql_integration.py` (temp sqlite external DB + mocked LLM: register → recall → entries; gate-off and non-owner-denied paths).

### PR 4 (optional, separable) — Agentic bridge
Create: `cognee/modules/tools/builtin/text_to_sql.py` (+ register in `cognee/modules/tools/builtin/__init__.py`): builtin `Tool` wrapping `run_text_to_sql`, handler re-checks gate + ownership, renders rows row-capped within `MAX_TOOL_OUTPUT_CHARS`. Tests: `cognee/tests/unit/modules/tools/test_text_to_sql_builtin.py` (arg validation, gate/ownership enforcement, reachability via `execute_tool` with dataset ACL).

---

## 8. Key tradeoffs accepted

- **No new `RecallTool` protocol** — one tool registry (`cognee/modules/tools/`), extensibility via the generic `ResponseToolEntry` + builtin registration; cost: recall's `_run_tools` hardcodes text-to-SQL dispatch in v1 (a tool-name switch is a contained later refactor).
- **`"tools"` opt-in only** — never in `"all"`/`"auto"`; cost: undiscoverable by default. Auto participation gated on "user has connections" is a possible follow-up.
- **Generic entry with `structured` dict** — tool #2 needs zero union/scope/runner changes; cost: weaker static typing of tool payloads. Note: strict typed clients validating the old union reject any new `source` literal — unavoidable for any new source.
- **New `ToolConnection` table** (vs overloading `IntegrationCredential`) — clean semantics; cost: second credential-bearing table sharing one crypto module.
- **Guard is UX, read-only transaction is load-bearing** — token/sqlglot guard gives fast rejection + retry feedback; the connection-level read-only layer stops anything that slips through.
- **Empty result = success** — saves LLM retries; a semantically-wrong zero-row query won't self-correct (the returned SQL makes it debuggable).
- **sqlite + postgres only in v1** — `create_relational_engine` raises on other providers even though dlt's `DB_CONNECTION_PATTERNS` accepts mysql/mssql/oracle strings; clear error, deferred extension.
- **Sequential runners preserved** — a slow external DB adds latency to multi-source recalls; statement timeout caps the worst case.

## 9. Open questions for the team

1. Which Linear issue (COG-xxx) does this land under? Branch must be renamed/recreated from `dev` with the key.
2. New `ToolConnection` table vs reusing `IntegrationCredential` with `provider="external_sql"` — plan picks the new table; confirm.
3. Is deployment-level `TOOL_SQL_CONNECTIONS` acceptable at all, or should v1 be per-user-registered only (multi-tenant deployments must not use the env path either way)?
4. Add `sqlglot` as a dependency for the guard, or ship the strict token fallback only?
5. Should v1 ever synthesize a natural-language answer over the rows (feeding results into an LLM)? Plan says no (prompt-injection-via-row-contents + latency); confirm.
6. Future auto-scope participation: include `"tools"` in `scope="auto"` when the user has registered connections?
7. Cross-user/team-shared connections: schedule the `ToolACL` follow-up (mirroring `give_permission_on_dataset`/`get_specific_user_permission_datasets`) now or wait for demand?
8. Cloud/serve: older servers 409 with an opaque message on `scope=["tools"]` — is a capability-discovery endpoint (or better error text) worth adding?
9. Prompt schema context: live `extract_schema()` with caps (v1) vs pre-ingested `SchemaTable` DataPoints via `ingest_database_schema` + vector-selected relevant tables (optimization) — timing?
