# Runtime configuration

See [`.env.template`](../.env.template) for the settings and their documented defaults.

## Important Configuration

### Environment Setup
Copy `.env.template` to `.env` and configure. Cognee finds the file once per process, at import, searching the working directory and its parents first, then the cognee package directory and its parents (`cognee/shared/env_file.py`). Each search stops at the project root, the nearest directory with a `.git` or `pyproject.toml`, so a `.env` above a project is never loaded; with no such marker (a plain folder of scripts) it continues up to the filesystem root. `COGNEE_ENV_FILE=/path/to/file` skips the search and loads that file (a path that is not a file raises at import). Values from the file take precedence over variables already set in the shell, for settings classes and `os.getenv` reads alike, so a project `.env` is the single source of truth wherever the environment happens to be installed. The path that was loaded (or "No .env file found") is logged at info on import.

```bash
# Minimal setup (defaults to OpenAI + local file-based databases)
LLM_API_KEY="your_openai_api_key"
LLM_MODEL="openai/gpt-6-luna"  # Default model
```

**Important**: If you configure only LLM or only embeddings, the other defaults to OpenAI. Ensure you have a working OpenAI API key, or configure both to avoid unexpected defaults.

**No key at all** is also a working setup: with no LLM and no embedding credentials configured, cognify extracts the graph with the local GLiNER demo model (`pip install "cognee[gliner]"` recommended; without it the CPU runtime installs itself on first use; see [LLM-free graph extraction](memory-internals.md#llm-free-graph-extraction-gliner-demo)) and embeds with a local model through fastembed, a core dependency (`perplexity-ai/pplx-embed-v1-0.6b`: a 0.7 GB download, 1024 dimensions, chunks capped at 512 tokens unless `EMBEDDING_MAX_COMPLETION_TOKENS` is set, a few chunks per second on a laptop CPU); models download on first use (a warning names the model, its size and the cache location while that happens; both models are kept in the Hugging Face hub cache, default `~/.cache/huggingface/hub`, which `HF_HOME` moves for both, with `FASTEMBED_CACHE_PATH` overriding it for fastembed alone), and `recall()` without a `query_type` routes only to search types that run without an LLM, defaulting to `HYBRID_COMPLETION` with `only_context=True` (the assembled prompt instead of an answer). The switch is per half — `GRAPH_EXTRACTOR=auto` (default) resolves on the LLM key, embeddings resolve on "nothing configured and no LLM key" — so setting any credential or any embedding setting takes that half back to the configured provider. The env vars that disable the preflight (`COGNEE_SKIP_PREFLIGHT`, `COGNEE_SKIP_CONNECTION_TEST`, `MOCK_EMBEDDING`) also disable this rerouting (`keyless_local_defaults_apply()`): a mocked or deliberately partial config is honoured, not replaced.

Default databases (no extra setup needed):
- **Relational**: SQLite (metadata and state storage)
- **Vector**: LanceDB (embeddings for semantic search)
- **Graph**: Ladybug (knowledge graph and relationships)

All stored in `.venv` by default. Override with `DATA_ROOT_DIRECTORY` and `SYSTEM_ROOT_DIRECTORY`.

### Switching Databases

#### Relational Databases
```bash
# PostgreSQL (requires postgres extra: pip install cognee[postgres])
DB_PROVIDER=postgres
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=cognee
DB_PASSWORD=cognee
DB_NAME=cognee_db
```

#### Vector Databases
Supported in-tree: lancedb (default), pgvector, neptune_analytics, turso.

**LanceDB payload schema migrations.** When a payload model changes (a field added or removed on `IndexSchema`, a type or nullability change), the stored `payload` struct is evolved in place by `cognee_db_workers/lancedb_schema_evolution.py` (its module docstring is the reference: Lance schema evolution on the payload column alone, vectors never read, `payload` present at every committed version, rows another process wrote during the run repaired by fragment, which is why compaction leaves a table mid-migration alone, interrupted runs finished by the next one, nothing partial). It runs proactively on every cognee version change (`run_migrations`, schema reads only, every table visited before one error names the failures) and reactively when a write hits LanceDB's schema-drift errors (`create_data_points`); in subprocess mode on the worker's handle (`OP_TABLE_EVOLVE_PAYLOAD`), with the model's required fields carried apart from the relaxed storage schema. Index collections are compared against `IndexSchema`, `TruthCentroid_vector` against its own model (`_resolve_collection_payload_schema`). A table whose vectors are not the configured width is refused with `StoredVectorWidthMismatchError` (a re-embedding, not a schema migration). Commits through pylance, so it needs the pylance line matching lancedb's Lance core, like compaction. Others (ChromaDB, Qdrant, Weaviate, Milvus, …) are community adapters — install from https://github.com/topoteretes/cognee-community and register via `use_vector_adapter` before setting `VECTOR_DB_PROVIDER`, otherwise cognee raises "Unsupported vector database provider".
**Re-embedding a dataset.** A dataset records the embedding model that built it (in its `dataset_database` row's `vector_database_connection_info`) and refuses any other (`EmbeddingDimensionMismatchError`, compared by vector width). `reembed_dataset(dataset_id)` (`cognee/modules/data/methods/reembed_dataset.py`) moves it to the deployment's configured embedding model, or to the one of an `embedding_config` passed to it, without re-ingesting: every collection of the dataset's vector store is rebuilt from the text its index rows hold (`cognee/tasks/storage/reembed_collections.py`), so the cost is one embedding per stored row and no LLM call, and the graph is not touched. Each collection is embedded into a staging collection first and replaced only once every one is staged, so a failure before that (the embedding provider, a row with no text) leaves the dataset as it was, with what was embedded so far discarded. From its start until it finishes the record carries the progress and every other entry into the dataset raises `DatasetReembedIncompleteError`; after an interrupted call, or one that failed while replacing, calling the function again with the same model continues where it stopped. An operation already inside the dataset is not stopped, so run it where nothing else uses the dataset. A dataset with a truth subspace is refused (`CogneeValidationError`): its centroids and the chunks' coordinates in the graph are computed from vectors and are not rebuilt. The dataset's routing rows are removed, and copied again (`copy_dataset_routes`) when the new model is the deployment's and the routing collection has its width; after an `EMBEDDING_MODEL` change that collection keeps the old width until it is dropped (`prune_dataset_routes`), so drop it before re-embedding the datasets. It works on LanceDB, pgvector and Turso (an adapter needs `get_collection_names` and `get_data_point_ids`), only with backend access control on (off, the datasets share one store), checks no permission, and has no SDK, CLI or HTTP surface: it is a utility for maintenance scripts.

**Local default embedding model change.** The local default was `BAAI/bge-small-en-v1.5` (384 dimensions) and is now `perplexity-ai/pplx-embed-v1-0.6b` (1024), which should provide better accuracy. The data migration `reembed_local_default_embedding_model` re-embeds, on upgrade, each dataset recorded as built with the previous default, and only where the local default is in force (no embedding setting, no usable LLM key). A configured model is used as it is and its datasets are left alone. To keep the previous model, set `EMBEDDING_PROVIDER=fastembed` and `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` before upgrading: the migration then has nothing to do. The migration runs with the others (server start, the first `remember()` or `cognify()`, `cognee-cli upgrade`), downloads the model on first use, and embeds on the CPU, so a large dataset takes a while and writes wait for it. A dataset it cannot re-embed (a truth subspace, a vector store other than LanceDB, pgvector and Turso, a row without text, a model that cannot be downloaded) is recorded as failed and writes to it are blocked; the error names the way out, which is that setting when the dataset is as it was and another run when it stopped while replacing collections. An interrupted re-embed is finished by the next run. If the previous model was set in between, a re-embed that had replaced nothing yet is undone, and one that had started replacing collections fails until it is finished without the setting. The migration reads the installation's settings, not a per-call `embedding_config`. An installation that kept the previous model and drops the setting later is not migrated again (the dataset is already stamped): call `reembed_dataset(dataset_id)` for each dataset then. Datasets whose record names no model (built before cognee 1.6.0) are left alone. With backend access control off the one shared store is not re-embedded: the migration raises with the same setting. `cognee-cli downgrade postgres_graph_source_ref_tables` re-embeds the datasets with the previous default, for going back to an older release. The migration runner enters a dataset without the embedding-model check for this (`runner._dataset_context`), and the re-embed it runs takes no dataset lock (`reembed_dataset_in_context`): the runner already holds the dataset's queue slot.
```bash
# PGVector (requires postgres extra)
VECTOR_DB_PROVIDER=pgvector
VECTOR_DB_URL=postgresql://cognee:cognee@localhost:5432/cognee_db
```

#### Graph Databases
Supported: ladybug (default), neo4j, neptune, ladybug-remote, turso (local file, requires the turso extra), postgres_demo (demo; `postgres` is an accepted alias)
```bash
# Neo4j (requires neo4j extra: pip install cognee[neo4j])
GRAPH_DATABASE_PROVIDER=neo4j
GRAPH_DATABASE_URL=bolt://localhost:7687
GRAPH_DATABASE_NAME=neo4j
GRAPH_DATABASE_USERNAME=neo4j
GRAPH_DATABASE_PASSWORD=yourpassword

# Remote Ladybug
GRAPH_DATABASE_PROVIDER=ladybug-remote
GRAPH_DATABASE_URL=http://localhost:8000
GRAPH_DATABASE_USERNAME=your_username
GRAPH_DATABASE_PASSWORD=your_password

# Postgres (requires postgres extra: pip install cognee[postgres])
# DEMO, not production-ready — see the warning below.
# Does not support raw Cypher queries or natural language search.
# The legacy value `postgres` still resolves to this same adapter.
GRAPH_DATABASE_PROVIDER=postgres_demo
GRAPH_DATABASE_URL=postgresql+asyncpg://cognee:cognee@localhost:5432/cognee_db
```

> **⚠️ Warning:** Using Postgres as a graph store is currently a demo feature and is not production-ready. Use it to demo keeping relational metadata, PGVector, and graph state in a single Postgres service, but rely on a graph-native backend such as Kuzu or Neo4j for production workloads.
>
> Interested in further development or production use of Postgres as a graph database? Write to us at social@cognee.ai to explore the options.

#### Turso

Turso supports local database files for the relational DB, graph store, vector store, and session cache. See [Turso setup](turso-local.md#setup) for provider settings, journal modes, and engine limits.

#### Session Cache
```bash
# Session/conversation cache backend: sqlite (default), turso, postgres, redis, tapes
CACHE_BACKEND=sqlite
# Optional explicit SQLAlchemy URL for sqlite/turso/postgres cache backends (overrides defaults)
CACHE_DB_URL=postgresql+asyncpg://cognee:cognee@localhost:5432/cognee_db
# Session-search execution mode: concurrent (default) or sequential
SESSION_SEARCH_MODE=concurrent
```

See [retrieval internals](retrieval-internals.md#session-search-modes) for session search modes and completion behavior.

### Memory & Performance Tuning Flags

Four flags trade memory features for speed. Know what each turns off before flipping it:

| Flag (default) | Turns off when disabled | Cost of disabling |
|---|---|---|
| `PERSONALIZATION_ENABLED=false` | Per-user preference personalization: one `UserPreference` node per user+dataset with weighted `prefers` edges, retrieval ranking multiplied by those weights, stated-preference text injected into LLM prompts, and the `improve()` stage that folds ratings into weights (the per-turn 1-5 rating question itself is part of automatic feedback analysis — gated by `AUTO_FEEDBACK`, not this flag — so the rating is produced and stored even before personalization is switched on) | Off by default, so nothing is lost until you opt in. When on, ranking strength comes from `PERSONALIZATION_INFLUENCE` (default 0.3, valid range [0, 1] — out-of-range values are rejected at startup); personalization also needs a user and a single resolved dataset in context, so multi-dataset searches never personalize |
| `CACHING=true` | The entire session-memory layer: `remember(session_id=...)` raises, `recall()` loses session history and the session-cache short-circuit, `agent_memory` session options error, and `AUTO_FEEDBACK` becomes moot | You lose the fast session write path and self-improving memory — only the slower add+cognify path remains. Do not benchmark cognee with this off; that measures cognee with its memory layer removed |
| `AUTO_FEEDBACK=true` | The automatic per-turn analysis: one structured-output LLM call after each answered query that detects implicit feedback, guides later retrievals, and feeds `improve()`'s agent-context lessons | Memory stops self-tuning from conversation signals. Session store/recall itself keeps working — this is the flag to disable for low-latency reads, since the per-turn LLM call dominates default read latency |
| `DATASET_QUEUE_ENABLED=true` | The per-process cap on concurrent datasets (`DATASET_QUEUE_MAX_CONCURRENT`, default 6), subprocess-engine release on scope exit (kept warm for `SUBPROCESS_IDLE_TTL_SECONDS`, default 600; closed at once when 0), and pinning of in-use engines against cache eviction. Only engages when `ENABLE_BACKEND_ACCESS_CONTROL` is on (its default) — with access control off the flag is a no-op either way | Saves minor per-operation overhead, but embedded engines become unbounded: file-lock leaks and mid-use engine eviction under parallel multi-dataset load. Safe only for single-dataset scripts |

`AUTO_FEEDBACK` is only consulted when `CACHING=true`. If reads feel slow on defaults, set `AUTO_FEEDBACK=false` and keep `CACHING=true` — that keeps session memory while removing the per-turn LLM call.

### LLM Provider Configuration

Supported providers: OpenAI (default), Azure OpenAI, Google Gemini, Anthropic, AWS Bedrock, Ollama, LM Studio, Custom (OpenAI-compatible APIs)

#### OpenAI (Recommended - Minimal Setup)
```bash
LLM_API_KEY="your_openai_api_key"
LLM_MODEL="openai/gpt-6-luna"  # default; or gpt-5.6-terra, gpt-5-mini, gpt-4o, etc.
LLM_PROVIDER="openai"
```

#### Azure OpenAI
```bash
LLM_PROVIDER="azure"
LLM_MODEL="azure/gpt-4o-mini"
LLM_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com/openai/deployments/gpt-4o-mini"
LLM_API_KEY="your_azure_api_key"
LLM_API_VERSION="2024-12-01-preview"
```

#### Google Gemini (no extra required)
```bash
LLM_PROVIDER="gemini"
LLM_MODEL="gemini/gemini-2.0-flash-exp"
LLM_API_KEY="your_gemini_api_key"
```

#### Anthropic Claude (requires anthropic extra)
```bash
LLM_PROVIDER="anthropic"
LLM_MODEL="claude-sonnet-5-5"
LLM_API_KEY="your_anthropic_api_key"
```

#### Ollama (Local - requires ollama extra)
```bash
LLM_PROVIDER="ollama"
LLM_MODEL="llama3.1:8b"
LLM_ENDPOINT="http://localhost:11434"  # bare host; /v1 only with the instructor framework
LLM_API_KEY="ollama"
EMBEDDING_PROVIDER="ollama"
EMBEDDING_MODEL="nomic-embed-text:latest"
EMBEDDING_ENDPOINT="http://localhost:11434/api/embed"
HUGGINGFACE_TOKENIZER="nomic-ai/nomic-embed-text-v1.5"
```

#### Custom / OpenRouter / vLLM
```bash
LLM_PROVIDER="custom"
LLM_MODEL="openrouter/deepseek/deepseek-r1"
LLM_ENDPOINT="https://openrouter.ai/api/v1"
LLM_API_KEY="your_api_key"
```
OpenRouter model ids change over time (the `:free` tier especially) — check `https://openrouter.ai/api/v1/models` for a current slug. Embeddings are a separate catalogue at `https://openrouter.ai/api/v1/embeddings/models` and must be configured separately; see the OpenRouter block in `.env.template`.

#### AWS Bedrock (requires aws extra)
```bash
LLM_PROVIDER="bedrock"
LLM_MODEL="anthropic.claude-3-sonnet-20240229-v1:0"
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your_access_key"
AWS_SECRET_ACCESS_KEY="your_secret_key"
# Optional for temporary credentials:
# AWS_SESSION_TOKEN="your_session_token"
```

#### Embedding worker over RabbitMQ

See the [embedding worker guide](embedding-worker.md) for setup, configuration, and failure behavior.

#### LLM Rate Limiting
```bash
LLM_RATE_LIMIT_ENABLED=true
LLM_RATE_LIMIT_REQUESTS=60  # Requests per interval
LLM_RATE_LIMIT_INTERVAL=60  # Interval in seconds
LLM_MAX_CONCURRENT_REQUESTS=0  # Max requests in flight at once per model; 0 = unlimited (default)
```
The RPM limiter bounds how many requests *start* per interval; `LLM_MAX_CONCURRENT_REQUESTS` bounds how many are *running* at once, which is what a serial backend (Ollama, llama.cpp, a small vLLM server) needs. It is off for every provider until set, applies per model (stages routed to the same model share its slots) and per process.

#### Instructor Mode (Structured Output)
```bash
# LLM_INSTRUCTOR_MODE controls how structured data is extracted
# Each LLM has its own default (e.g., gpt-4o models use "json_schema_mode")
# Override if needed:
LLM_INSTRUCTOR_MODE="json_schema_mode"  # or "tool_call", "md_json", etc.
```

### Structured Output Framework
```bash
# litellm_native (default): plain litellm, schema-native response_format
# with prompted-JSON fallback — no instructor in the call path
STRUCTURED_OUTPUT_FRAMEWORK="litellm_native"

# Or use Instructor (legacy, via litellm; requires instructor extra: pip install cognee[instructor])
STRUCTURED_OUTPUT_FRAMEWORK="instructor"

# Or use BAML (requires baml extra: pip install cognee[baml])
STRUCTURED_OUTPUT_FRAMEWORK="baml"
BAML_LLM_PROVIDER=openai
BAML_LLM_MODEL="gpt-4o-mini"
BAML_LLM_API_KEY="your_api_key"
```

### Storage Backend
```bash
# Local filesystem (default)
STORAGE_BACKEND="local"

# S3 (requires aws extra: pip install cognee[aws])
STORAGE_BACKEND="s3"
STORAGE_BUCKET_NAME="your-bucket-name"
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your_access_key"
AWS_SECRET_ACCESS_KEY="your_secret_key"
DATA_ROOT_DIRECTORY="s3://your-bucket/cognee/data"
SYSTEM_ROOT_DIRECTORY="s3://your-bucket/cognee/system"
```

## Security Considerations

Several security environment variables in `.env`:
- `ACCEPT_LOCAL_FILE_PATH` - Allow a path in `raw_data` to be read with the server's own access: a local file or directory, cognee's own data root included, or an `s3://` object read with cognee's AWS credentials. One gate, `local_file_paths_accepted()` in `cognee/tasks/ingestion/save_data_item_to_storage.py`, applied before anything is read or listed (`resolve_data_directories` refuses a directory or S3 prefix before walking it): an explicit value is honoured on every surface; unset, a local caller (the SDK, the CLI, the MCP server over stdio) keeps the code default (True) while a network caller is refused: every API request, and every MCP tool call served over `sse` or streamable `http`, whatever the bind address. The gate reads the remote-caller signal (`cognee/modules/operations/remote_caller.py`), set by the API middleware per request and by the MCP server's `RemoteCallerMiddleware`; a task a request starts keeps it. It does not read the operation origin, which stays `api` / `mcp` for `pipeline_runs` and telemetry. With the gate closed, `add()` decides each caller string once, at its entry, by shape alone (`screen_server_paths` in `cognee/tasks/ingestion/screen_server_paths.py`, also run by the dry-run estimator and the MCP client's API-mode upload): a web URL or connection string goes on, a server path (`file://`, `s3://`, or a string that is only a path: one token that starts with a root, `~`, `./`, `../` or a drive and holds a separator after its first character, such as `/etc/passwd` or `~/x`, or a root hidden file such as `/.env`, while `/start`, `CI/CD` and `docs/../x` are text) is refused whether or not it exists, and anything else becomes literal text, so no later step can look it up and no reply reveals whether a path exists. There is no exemption for cognee's storage: a caller's path into it is refused like any other. The documents of a repository cognee cloned from a URL are the one path form that passes without the gate: `resolve_data_directories` marks them `DataItem.cognee_path`, since the caller named a URL and cognee produced the paths. Uploads, text and web URLs are unaffected: they are stored under cognee's data root first and loaded from there, and the loader (`data_item_to_text_file`) applies no policy of its own, since everything it receives is cognee's copy or already admitted by the gate. `PATCH /update` takes the new version as an upload, so a document in a dataset shared with the caller is updated by upload under the dataset's write permission alone
- `COGNEE_ALLOWED_LOCAL_FILE_ROOTS` - Optional `os.pathsep`-separated allowlist of directories local paths may be read from, applied once a path is accepted at all (over HTTP that needs `ACCEPT_LOCAL_FILE_PATH=true`). Unset (default) means any accepted local path is read, so a local repo or document tree can be ingested from anywhere; with the gate open, a path-looking string that does not exist is still ingested as text (closed, a path-only string is refused, see above). When set, paths outside the listed roots are rejected (or ingested as text on the non-strict `add()` path); cognee's own data/system/cache/logs/repos roots are always allowed. Set it when paths are accepted from untrusted callers.
- `ALLOW_HTTP_REQUESTS` - Allow outbound URL fetching and remote repository cloning. Unset: disabled for API requests, enabled for SDK/CLI/MCP calls. Set `true` in the server environment to enable URL ingestion over the API; a set value overrides the default on every surface. Only `true`, `1`, `yes` or `on` enable it; any other value (`false`, or a misspelling) disables it, and a blank value counts as unset (`opt_in_setting` in `cognee/tasks/web_scraper/ssrf_protection.py`, shared with `ALLOW_DATABASE_INGESTION`). The refusal is HTTP 403 and names the setting; `add()` and `remember()` check their inputs before any work starts (`require_http_requests_allowed_for`), so a `run_in_background=True` call is refused in its response instead of starting a run that fails later. Connectors (Slack, Notion, Google, Linear, the GitHub App) are not under the API default, since they only call their providers' fixed hosts: they work unset or blank and stop only when a set value disables it (`require_outbound_http` in `cognee/modules/integrations/outbound.py`); the GitHub App's clones of the `github.com` URLs it builds from GitHub's own repository names run as `background` work for the same reason. This does not disable LLM or embedding provider calls.
- `ALLOW_DATABASE_INGESTION` - **Database ingestion by network callers is off by default.** A `postgresql://`, `mysql://`, `sqlite:///` (any `DB_CONNECTION_PATTERNS` scheme) string posted to `POST /api/v1/add` or `POST /api/v1/remember`, or passed to an MCP tool served over `sse` or streamable `http`, is refused with 422 `DatabaseIngestionDisabled` unless the server sets `ALLOW_DATABASE_INGESTION=true` (a change from releases before 1.6.6, which accepted it); the SDK, CLI and MCP server over stdio ingest databases with the variable unset. The gate reads the same remote-caller signal as `ACCEPT_LOCAL_FILE_PATH`. The reason: the connection is made from the server, with its network position, and a network caller is whoever can reach it. `false` refuses connection-string sources on every surface, and so does any other set value that is not `true`, `1`, `yes` or `on` (a misspelling fails closed); a blank value counts as unset. Connection strings that point at cognee's own storage are refused whatever the setting: SQLite files inside its data/system/cache/logs/repos roots or configured database paths, and network databases on a server its relational, vector, graph or cache settings use, matched by server and not by database name (the per-dataset Postgres handlers keep every dataset's database on one server). A server is recognised two ways. By credentials: a string carrying one of cognee's own username/password pairs (the relational, vector and graph settings' fields and the vector, graph and cache URLs) is refused on any host, which needs no address and so holds under aliases the address check cannot know and under DNS rebinding; it cannot cover a server that admits clients without a password or another role with access to cognee's databases. By address: the host resolved and compared with cognee's servers' addresses on the same port, where every loopback address, `0.0.0.0`/`::`, an IPv4-mapped IPv6 address and whatever this machine's hostname resolves to count as this machine. The address check is best-effort: it cannot list every address of this machine (on Debian-family hosts the hostname resolves to `127.0.1.1` alone) or of the database host (the host's LAN address behind the compose default `DB_HOST=host.docker.internal`), so an operator who turns the switch on relies on the credential refusal, on cognee's databases not accepting passwordless clients, and on network egress rules. A `sqlite:///` path must be an existing regular file (the driver would otherwise create it) and may not carry the `uri` option. A network connection string must name its server as the URL host and may carry only the `sslmode`, `connect_timeout`, `application_name` and `charset` options. Driver errors are reported by exception class only (`DatabaseSourceUnreachable`), with the full error in the server log.
- `ALLOW_CYPHER_QUERY` - Allow raw Cypher queries, the CYPHER and NATURAL_LANGUAGE search types (default: False). Off by default because raw Cypher lets any user with read access run arbitrary queries, up to remote code execution; enable only when all users are trusted
- `FILE_ENCRYPTION_ENABLED` - Enable application-layer encryption for files under `DATA_ROOT_DIRECTORY` (default: `false`). See [File encryption](file-encryption.md) for setup and scope.
- `COGNEE_LOG_SEARCH_HISTORY` - Write every answered `search()` and `recall()` call as plaintext question and answer rows to the relational database (default: True). Read at call time. Set it to `false` when question or answer text must stay out of the history, since answers can quote source chunks. See [search history retention and deletion](memory-internals.md#search-history-retention-and-deletion).
- `ENABLE_BACKEND_ACCESS_CONTROL` - Multi-tenant isolation (default: True). When `true`, API auth is required and per-user/dataset DB isolation is enabled. When `false`, single-user mode: shared DBs and auth off unless overridden.
- `REQUIRE_AUTHENTICATION` - Explicit auth override. Unset (default): follows `ENABLE_BACKEND_ACCESS_CONTROL`. `false` is ignored when `ENABLE_BACKEND_ACCESS_CONTROL=true`. For a single-user deployment with auth off, set `ENABLE_BACKEND_ACCESS_CONTROL=false` (and optionally `REQUIRE_AUTHENTICATION=false`).
- `AUTH_TOKEN_COOKIE_SECURE` - `Secure` attribute on the auth cookie the login route writes (`true` or `false`, default false; anything else raises at import). The cookie is the session; `Secure` stops a browser sending it over plain http. Set it to `true` on any deployment served over TLS, including behind a TLS-terminating proxy, since cognee itself sees plain http there and cannot infer it. Leave it `false` only for `http://localhost`, where a browser will not store a `Secure` cookie.
- `REFRESH_TOKEN_LIFETIME_SECONDS` - How long a browser login lasts without being used (default: 604800, 7 days). An access token is valid for `JWT_LIFETIME_SECONDS` (default 3600) and cannot be extended, so the login route also sets a refresh cookie (the auth cookie's name with `_refresh` appended) that `POST /api/v1/auth/refresh` exchanges for a new access token; the web UI does this on its own when a request returns 401. Each use restarts the time. The cookie has the auth cookie's attributes except its path, which is the path of the login, refresh and logout routes (`/api/v1/auth`), so the browser sends the token to those routes only. The path is taken from the login request as the server sees it: behind a reverse proxy that serves the API under a path prefix, start the server with that prefix (`uvicorn --root-path`), or the browser will not send the cookie and logins will not be renewed. The token is stored hashed in the relational database: `POST /api/v1/auth/logout` deletes it, and a password change or an inactive user makes it unusable. It is also tied to `FASTAPI_USERS_JWT_SECRET`: changing the secret signs every browser out, and with the secret unset (a random one per process) every restart does. Bearer token and API key clients are unaffected: the refresh token is only ever sent as a cookie.
- `REFRESH_TOKEN_MAX_LIFETIME_SECONDS` - How long a browser login lasts in total, however often it is used (default: 2592000, 30 days). `REFRESH_TOKEN_LIFETIME_SECONDS` restarts with every use, so on its own a copied refresh cookie could be kept alive for good by using it once a week; this setting ends the login a fixed time after it started, and the user logs in again. Both settings have to be a positive whole number of seconds: the server does not start with any other value.

For production deployments, review and tighten these settings.

## Debugging & Troubleshooting

### Debug Configuration
- Set `LITELLM_LOG="DEBUG"` for verbose LLM logs (default: "ERROR")
- Enable debug mode: `ENV="development"` or `ENV="debug"`
- Disable telemetry: `TELEMETRY_DISABLED=1`. It also turns off the telemetry dlt sends to its own endpoint during dlt ingestion (connectors, database and CSV sources), by defaulting `RUNTIME__DLTHUB_TELEMETRY=false`; cognee also defaults `LOAD__DELETE_COMPLETED_JOBS=true`, so dlt keeps no copy of loaded rows in its load packages. Both are environment defaults set before cognee builds a dlt source or pipeline (`apply_dlt_runtime_defaults` in `cognee/tasks/ingestion/dlt_utils.py`) and apply to the whole process, a host application's own dlt pipelines included: a value already in the environment wins, one set only in dlt's `config.toml` does not, and a process that used dlt before cognee's first dlt call (a `@dlt.source` or `sql_database()` the caller built) has already started dlt's telemetry, so set `RUNTIME__DLTHUB_TELEMETRY=false` yourself there. A failed load keeps its package until the same pipeline runs again. `drop_dlt_dataset` (`cognee/tasks/ingestion/ingest_dlt_source.py`) deletes what dlt keeps of one dataset, its staging database and pipeline directories; `forget()` does not call it.
- Telemetry events use the existing operation-origin context: `sdk`, `cli`, `mcp`, `api`, or `background`. An explicit `TELEMETRY_ORIGIN` (e.g. `cloud`) overrides it. `TELEMETRY_ORIGIN=cloud` (set by Cognee Cloud for every tenant pod, in the `cognee-tenant-operator` Helm charts in cloud-backend) selects the cloud profile, a versioned metadata-only contract (`cognee/shared/telemetry_schema.py`, `CLOUD_SCHEMA_VERSION`; policy CLO-799; field list in `docs/cloud-telemetry-contract.md`): an event leaves only when `CLOUD_EVENT_MODELS` maps it to a strict Pydantic model (run milestones and memory/search outcomes), and only with the fields that model declares; an undeclared field or a value that fails its check is removed alone, the rest of the event still goes, and each removal is counted in `CLOUD_REJECTIONS` by event and field path, never by value. Identity and environment (`user_id`, `tenant_id`, agent fields, `schema_version`, `telemetry_origin`) are built by `send_telemetry` only (`cognee_version` stays the emitter's own value: recovery reports `unknown`), validated too (a bad one drops the event), and no event property can override them. A cloud event carries no dataset names, session ids or their fingerprints, database URLs, `anonymous_id` / `persistent_id` (`anonymous_id` is sent as `""`, a key the proxy requires), API-key hash or `connected_*`. Any failure in the contract drops the event and never reaches the caller. Every other origin keeps the full payload, unchanged.
- Events are sent in batches by one background thread (`cognee/shared/telemetry_sender.py`). Each event is encoded when queued: one that is not JSON or exceeds 16 KiB is dropped alone. The queue keeps at most 10,000 events / 1 MiB (oldest dropped first), and a request carries at most 100 events / 64 KiB. Network errors and 5xx are retried; a 4xx drops the batch, never retried or split into single events. Rate limits are the proxy's job.
- Every event carries `is_agent`, `agent_owner_id` and `agent_plugin`, read off the `User` passed to `send_telemetry` (`cognee/modules/integrations/agent_identity.py`). An agent is a child user with its own API key, such as a provisioned plugin, so its `user_id` is the agent itself; `agent_owner_id` is the owning user and `agent_plugin` a `KNOWN_PLUGINS` key or `"custom"` (an agent's free-text name is never sent). A bare id cannot be told apart from a human's, so emitters pass the model, not `user.id` (a unit test enforces it). API error events take the caller `get_authenticated_user` stores on `request.state.user`; a deployment that overrides `get_authenticated_user` must store it too, or its error events carry no user. The same dependency calls `bind_trace_owner(user)` (`cognee.modules.observability`), which marks the request's spans in the in-memory trace buffer as that user's; `GET /api/v1/activity/spans` returns only traces owned by the caller and their own agents, and none without an owner, so an override that skips the call leaves that endpoint empty (never shared).
- `Pipeline Run` events follow the dataset lifecycle in `run_tasks`; completion is emitted only after durable storage and the run record succeed. They carry `pipeline_event_scope=run`. `Pipeline Item` events follow individual items and add their loader, extension, size and token buckets (`data_item_telemetry_properties`). Both carry `pipeline_run_id` and the provider stack from `get_current_settings()`, including structured-output settings and cognify's already-resolved extractor. Historical item-scoped `Pipeline Run` events cannot establish whole-run completion; the aggregate reports keep them separate.
- Failed `search()`/`recall()` calls emit `ERRORED`. Pipeline, item and task failures use `telemetry_exception_properties` (`cognee/shared/utils.py`): exception class (unwrapping `first_error`), cause chain (at most four links), and the innermost HTTP status, never the message. Model settings that are filesystem paths are reported as `local_path`.
- Check logs in structured format (uses structlog)
- Use `debugpy` optional dependency for debugging: `pip install cognee[debug]`

### Common Issues

**Slow search/recall on default settings**
- Issue: Each answered query on the session path makes one structured-output LLM call for automatic feedback analysis
- Solution: Set `AUTO_FEEDBACK=false` (keep `CACHING=true` so session memory stays on); see [memory and performance tuning flags](#memory--performance-tuning-flags)

**Ollama + OpenAI Embeddings NoDataError**
- Issue: Mixing Ollama with OpenAI embeddings can cause errors
- Solution: Configure both LLM and embeddings to use the same provider, or ensure `HUGGINGFACE_TOKENIZER` is set when using Ollama

**LM Studio Structured Output**
- Issue: LM Studio requires explicit instructor mode
- Solution: Set `LLM_INSTRUCTOR_MODE="json_schema_mode"` (or appropriate mode)

**Default Provider Fallback**
- Issue: Configuring only LLM or only embeddings defaults the other to OpenAI
- Solution: Always configure both LLM and embedding providers, or ensure valid OpenAI API key

**Permission Denied on Search**
- Behavior: Without `datasets`, search covers only datasets the user can read, so with `ENABLE_BACKEND_ACCESS_CONTROL` on (default) a user without grants gets an empty list (with it off, search runs over the shared graph). An explicit dataset id the user cannot read raises `PermissionDeniedError` (HTTP 403). Dataset names resolve only among the user's own datasets, so a shared dataset's name raises `DatasetNotFoundError`; pass its id instead.
- Solution: Check dataset permissions and user access rights

**Database Connection Issues**
- Check: Verify database URLs, credentials, and that services are running
- Docker users: Use `DB_HOST=host.docker.internal` for local databases

**Rate Limiting Errors**
- Enable client-side rate limiting: `LLM_RATE_LIMIT_ENABLED=true`
- Adjust limits: `LLM_RATE_LIMIT_REQUESTS` and `LLM_RATE_LIMIT_INTERVAL`

**Local LLM server times out during ingestion**
- Issue: cognify sends one extraction call per chunk at once, so a serial server (Ollama, llama.cpp, a small vLLM deployment) queues them until callers time out and retry. The RPM limiter does not help: it limits how many calls start per interval, not how many are running
- Solution: set `LLM_MAX_CONCURRENT_REQUESTS` to what the server runs in parallel (Ollama's `OLLAMA_NUM_PARALLEL`), usually 1 to 4. It applies per model and per process, so N workers can send N times that
