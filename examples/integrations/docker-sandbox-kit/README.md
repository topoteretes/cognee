# Cognee memory kit for Docker Sandboxes

A [Docker Sandboxes kit](https://docs.docker.com/ai/sandboxes/customize/kits/)
that gives any sandboxed coding agent persistent, self-improving memory backed
by [cognee](https://github.com/topoteretes/cognee) memory layer. Everything runs embedded inside the sandbox

Verified end-to-end with sbx v0.39.0 under a `deny-all` network policy.

## What the kit does

`cognee-memory/spec.yaml` is a stackable **mixin kit** that:

- installs the `cognee-cli` with `uv tool install cognee` at sandbox creation;
- pins all memory state to `/home/agent/.cognee`, so it survives sandbox
restarts;
- allowlists only what cognee actually needs under `deny-all` —
`api.openai.com`, PyPI (install), `extension.ladybugdb.com` (the embedded
graph DB fetches its extensions at first use), and
`raw.githubusercontent.com` (litellm's model-cost map). The list was derived
by running under `deny-all` and reading `sbx policy log`;
- declares a proxy-managed credential (`cognee-openai` — deliberately not
`openai`, which the built-in agent kits already declare; two kits defining
the same service fail composition). The agent only ever sees a placeholder
value; the sandbox proxy injects the real key in transit;
- pins `ENABLE_BACKEND_ACCESS_CONTROL=true` (cognee's default): multi-tenant
ACLs and per-user+dataset database isolation;
- appends usage instructions to the agent's memory file
(`kits-memory/cognee-memory.md`): `recall` at task start → work →
`remember` durable learnings, plus the multi-agent handover pattern.

## Setup (one-time)

```console
$ brew trust docker/tap && brew install docker/tap/sbx
$ sbx daemon start          # own terminal, or: nohup sbx daemon start &
$ sbx login                 # browser OAuth
$ sbx policy init deny-all  # strictest baseline; the kit's allowlist is the only egress
$ sbx secret set-custom --host api.openai.com --env LLM_API_KEY --value "$LLM_API_KEY"
```

`set-custom` prints a placeholder (`sbx-cs-…`, retrievable via `sbx secret ls`).
Sandboxes only ever see the placeholder; the proxy substitutes the real key on
requests to `api.openai.com`.

## Single-agent usage

```console
$ sbx run claude --kit ./cognee-memory          # agent with persistent memory
$ sbx run shell  --kit ./cognee-memory          # or a plain shell sandbox
```

Inside, the agent has `cognee-cli remember / recall / improve / forget`. For
headless `sbx exec` use, set the key to the placeholder explicitly (the kit's
`proxy-managed` env value is for the interactive credential-binding path):

```console
$ sbx exec <sandbox> -- sh -lc 'export LLM_API_KEY=<placeholder> LOG_LEVEL=ERROR; \
    cognee-cli remember "fact worth keeping"'
```

## Multi-agent demo: supervisor → worker memory handover

`./demo/handover.sh` runs a **round-trip handover between two real
sandboxes**. Both are created from this kit and share the `demo/` directory as
their workspace. The cognee state runs on each VM's **local disk** during a
phase (embedded LanceDB cannot operate on the shared virtiofs workspace
mount — discovered the hard way) and is handed between sandboxes as a
snapshot with `sbx cp`, making the memory handover literal. The host keeps
the canonical snapshot in `demo/cognee-state/` between phases. Even though
the worker receives the whole snapshot, the supervisor and worker are
separate cognee **users**, so ACLs still gate what each can read or write:

1. **brief** (`cognee-supervisor` sandbox): stores a private note and a
 handover briefing in its own datasets, grants the worker **read + write**
 on the briefing with `authorized_give_permission_on_datasets(...)` (the
 creator automatically holds `share`), and writes the handover token
 (`demo/handover-out/handover_token.json`) carrying the dataset UUID.
2. **work** (`cognee-worker` sandbox): redeems the token —
 `cognee.recall(..., dataset_ids=[uuid], user=worker)`. Sharing works
 **only by UUID**: dataset names are namespaced per user
 (`uuid5(name + user.id + tenant_id)`), so a name never crosses a user
 boundary. Negative checks: the private dataset raises
 `PermissionDeniedError` (403); the shared dataset by name fails (404).
 Then it writes its completion report back into the shared dataset
 (`cognee.remember(..., dataset_id=uuid, user=worker)`).
3. **review** (`cognee-supervisor` sandbox): recalls the worker's report.

Phases run sequentially — the snapshot moves, it is never shared live. The
payload, `demo/supervisor_worker_handover.py`, is
self-contained: paste it into any repository with cognee installed and run
`python supervisor_worker_handover.py` (all phases in-process) or
`--phase brief|work|review` split across environments.

```console
$ export LLM_API_KEY=sk-...     # only needed the first time, for the secret
$ ./demo/handover.sh
$ sbx policy log                # the audit trail: per-domain allow/deny
```

Cleanup: `sbx rm -f cognee-supervisor cognee-worker && rm -rf demo/cognee-state demo/handover-out`

### User permissioning: what currently supports it

- Permissions are `read` / `write` / `delete` / `share` per dataset. Managing
users and grants is **Python-SDK/REST-only** today — `cognee-cli` has no
user/permission commands.
- Backends supporting per-user+dataset DB isolation (source of truth:
`supported_dataset_database_handlers.py`): graph — kuzu/ladybug (default),
Neo4j (multi-db editions, plus a `neo4j_community` container-per-dataset
handler), Postgres (demo), Turso; vector — LanceDB (default), PGVector,
Turso. **Not** supported: Neptune, ladybug-remote, Neptune Analytics, and
community vector adapters unless they register a dataset-database handler.

## Inspecting memory and security

```console
$ sbx exec cognee-supervisor -- sh -lc 'echo $LLM_API_KEY'   # "proxy-managed" — never a real key
$ sbx exec cognee-supervisor -- curl -s -o /dev/null -w "%{http_code}" https://example.com   # 403: deny-all
$ sbx policy log                                             # every allow/deny decision
$ ls demo/cognee-state/system/databases/<owner-user-uuid>/   # <dataset-uuid>.lbug + .lance.db per dataset
```

The relational DB (`demo/cognee-state/system/databases/cognee_db`, SQLite)
holds users, datasets, and the ACL rows — after the demo, the worker holds
exactly two grants (`read`, `write`) on the shared dataset and nothing on the
private one.

## Layout

```
docker-sandbox-kit/
├── cognee-memory/         # the kit — point --kit here
│   └── spec.yaml
├── demo/
│   ├── handover.sh        # 2 real sandboxes, permissioned round-trip handover
│   ├── handover-out/      # the JSON handover token (created at runtime)
│   ├── cognee-state/      # canonical memory snapshot between phases (runtime)
│   └── supervisor_worker_handover.py
└── README.md
```

## Notes

- `remember` builds a knowledge graph (a few LLM calls), so the first write
takes noticeably longer than a plain key-value store; `recall` answers from
the graph.
- The kit defaults to `openai/gpt-5-mini`. To use another provider, edit
`environment.variables`, the `credentials`/`permissions.network` blocks, and
the stored secret accordingly (see the [cognee provider docs](https://docs.cognee.ai/)).
- For always-on cross-sandbox memory (concurrent agents, no shared workspace),
run a central cognee API server and point sandboxes at it over the network
allowlist instead of sharing embedded storage.
