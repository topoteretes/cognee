# Running this fork on Railway

This repo is a fork of [topoteretes/cognee](https://github.com/topoteretes/cognee).
`main` tracks an upstream release tag plus the small overlay listed here; `dev`
is `main` plus whatever is under test. Dev services deploy `dev`, prod deploys `main`.

## Overlay (what differs from upstream)

- `railway.api.toml` — API service: builds `Dockerfile.railway`, `/health` check.
- `Dockerfile.railway`, `scripts/railway-dockerfile.sh` — upstream's Dockerfile minus BuildKit cache mounts, which Railway rejects without an id. Rerun the script after every upstream sync.
- `railway.mcp.toml`, `Dockerfile.mcp` — MCP service, built on the published `cognee/cognee-mcp` image.
- `docs/railway.md` — this file.

Everything else is upstream. Keep it that way: prefer an env var or a file in this
list over patching an upstream file, because every patched upstream file is a
conflict on the next sync.

## Service variables the image needs

Beyond the database, LLM and embedding settings already on the services:

- `COGNEE_EXTRAS=docs` — build arg; installs the loaders for docx, pptx, xlsx and pdf.
- `HTTP_PORT=${{PORT}}` — upstream's entrypoint binds `HTTP_PORT`, Railway provides `PORT`.
- `SYSTEM_ROOT_DIRECTORY=/data/system`, `DATA_ROOT_DIRECTORY=/data/data` — the volume. The image runs as uid 1000; the mount must be writable by it.
- `KUZU_MAX_DB_SIZE` — optional, defaults to 128 GiB per dataset store (power of two).
- `GRAPH_DATABASE_PROVIDER=kuzu`, `GRAPH_DATASET_DATABASE_HANDLER=kuzu` — unchanged; the engine package is now `ladybug`.

## Syncing with upstream

```
git fetch upstream --tags
git checkout dev
git merge vX.Y.Z        # merge, never rebase: main is deployed
```

Deploy `dev`, verify, merge `dev` into `main`.

## Upgrading stores from the old template (kuzu 0.11.3)

On first open, `cognee_db_workers/ladybug_migrate.py` exports each kuzu 0.11.3 store
(storage code 39) and imports it into ladybug, keeping an `_old` copy beside it.
Rehearse on dev with a copy of the prod volume before touching prod.
