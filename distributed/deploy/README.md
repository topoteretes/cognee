# Cognee Deployment

1-click deployment configurations for hosting Cognee as a service.

## Quick Start

| Platform | Best For | Command |
|----------|----------|---------|
| **Modal** | Serverless, auto-scaling, GPU workloads | `bash distributed/deploy/modal-deploy.sh` |
| **Railway** | Simplest PaaS, native Postgres | `railway init && railway up` |
| **Fly.io** | Edge deployment, persistent volumes | `bash distributed/deploy/fly-deploy.sh` |
| **Render** | Simple PaaS with managed Postgres | Deploy to Render button |
| **Daytona** | Cloud sandboxes (SDK or CLI) | `python distributed/deploy/daytona_sandbox.py` |
| **Islo** | Isolated cloud sandboxes for agents (SDK) | `python distributed/deploy/islo_sandbox.py` |

All platforms require setting `LLM_API_KEY` as a minimum.

---

## Modal (Serverless)

Best for bursty workloads — scales to zero when idle, auto-scales under load. No infrastructure to manage.

```bash
# Install Modal CLI
pip install modal && modal setup

# Deploy (set your API key first)
export LLM_API_KEY=sk-xxx
bash distributed/deploy/modal-deploy.sh
```

The script creates a Modal secret group and deploys the FastAPI server. Your endpoint URL will be shown in the Modal dashboard.

**Configuration**: Edit `distributed/deploy/modal_app.py` to adjust:
- `timeout` — max request duration (default: 3600s for long cognify jobs)
- `container_idle_timeout` — time before scaling to zero (default: 300s)
- `allow_concurrent_inputs` — requests per container (default: 10)

**Persistent data**: Uses a Modal Volume mounted at `/data` for file-based databases. For production, configure Postgres + PgVector instead.

### End-to-end test

`tests/test_modal_serve_e2e.py` verifies the serving path against real Modal
infrastructure: it deploys an ephemeral, uniquely named app, polls `/health`
(tolerating the scale-to-zero cold start), runs a minimal add → cognify →
search round-trip over HTTP, and always tears the app and volume down —
even on failure. Because the Modal image installs the published `cognee`
package, this tests the released serving path, not local branch code.

In CI it runs via `.github/workflows/modal_serve_e2e.yml` — nightly and on
manual `workflow_dispatch` only, never on PRs — and skips cleanly when the
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` secrets are absent (e.g. on forks).

To run it locally with a Modal account and the `cognee-secrets` group set up:

```bash
python distributed/deploy/tests/test_modal_serve_e2e.py
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODAL_APP_NAME` | `cognee-api-e2e-<run id>` | Ephemeral app name (production name is refused) |
| `MODAL_VOLUME_NAME` | `cognee-data-e2e-<run id>` | Ephemeral volume name, deleted on teardown |
| `DEFAULT_USER_EMAIL` / `DEFAULT_USER_PASSWORD` | cognee defaults | Default-user login; must match the values in `cognee-secrets`, if set there |

---

## Railway

Simplest path to a hosted Cognee API. Native Postgres add-on with pgvector support.

### Option A: Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli && railway login

# From the cognee repo root:
cp distributed/deploy/railway.toml .
railway init
railway up
```

### Option B: 1-Click Template
Use the Railway template in `distributed/deploy/railway-template.json` to create a "Deploy on Railway" button. The template provisions:
- Cognee API service (from Dockerfile)
- PostgreSQL with pgvector
- Auto-wired environment variables

**Cost**: ~$5/mo hobby tier.

---

## Fly.io

Edge deployment with persistent volumes. Good latency for global users.

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh && fly auth login

# Deploy
export LLM_API_KEY=sk-xxx
bash distributed/deploy/fly-deploy.sh
```

The script handles app creation, secrets, volume provisioning, and deployment. Your API will be at `https://cognee.fly.dev`.

**Customization**: Edit `distributed/deploy/fly.toml` to adjust:
- `primary_region` — deployment region
- `vm.memory` / `vm.cpus` — instance sizing
- `auto_stop_machines` — set to `"off"` to keep always-on

---

## Render

Simple PaaS with managed Postgres and persistent disks.

### Deploy with Blueprint
The `distributed/deploy/render.yaml` blueprint provisions:
- Cognee API web service
- PostgreSQL 17 database
- 10GB persistent disk for file-based data

```bash
# Copy render.yaml to repo root and push
cp distributed/deploy/render.yaml render.yaml
git add render.yaml && git commit -m "Add Render blueprint"
git push
```

Then connect the repo in the Render dashboard and deploy.

---

## Daytona (Cloud Sandbox)

Daytona provides secure, isolated cloud sandboxes. Cognee runs inside a sandbox with persistent storage.

### Option A: Python SDK
```bash
pip install daytona

export DAYTONA_API_KEY=your-key   # from https://app.daytona.io
export LLM_API_KEY=sk-xxx
python distributed/deploy/daytona_sandbox.py
```

### Option B: CLI
```bash
brew install daytonaio/cli/daytona
daytona create
# Inside the sandbox:
pip install 'cognee[api]'
python -m uvicorn cognee.api.client:app --host 0.0.0.0 --port 8000
```

---

## Islo (Cloud Sandbox)

Islo provides isolated cloud sandbox VMs for autonomous agents, built by the Incredibuild team. Cognee runs inside a sandbox and is exposed through a temporary public share URL. Docs: https://docs.islo.dev

```bash
curl -fsSL https://islo.dev/install.sh | bash
islo login
islo api-key create cognee-deploy --expires 90 --show

pip install islo
export ISLO_API_KEY=your-cli-created-key
export LLM_API_KEY=sk-xxx
python distributed/deploy/islo_sandbox.py
```

The CLI is used only to create the access key. The deployment itself uses the official Python SDK to create the sandbox, install `cognee[api]`, start the API server, verify `/health`, and create a 24-hour share URL. Stop or delete the sandbox via the SDK:

```python
from islo import Islo

client = Islo()  # reads ISLO_API_KEY from the environment
client.sandboxes.stop_sandbox("cognee-api")
client.sandboxes.delete_sandbox("cognee-api")
```

The sandbox name is fixed (`cognee-api`), so re-running the script while a previous deployment still exists fails with a name conflict — delete the old sandbox first (see above), then re-run.

---

## Devcontainers (Codespaces / VS Code)

For contributors who want a pre-configured development environment. Uses `.devcontainer/devcontainer.json` at the repo root.

### GitHub Codespaces
```bash
gh codespace create --repo topoteretes/cognee
```

### VS Code Dev Containers
Open the repo in VS Code and select "Reopen in Container".

---

## Docker Compose (Self-Hosted)

For running on your own infrastructure, use the existing docker-compose setup:

```bash
# Minimal (SQLite + LanceDB + Ladybug - no external deps)
docker-compose up cognee

# With Postgres + pgvector
docker-compose --profile postgres up

# With Neo4j graph database
docker-compose --profile neo4j up

# Full stack with UI
docker-compose --profile ui up
```

---

## Production Recommendations

1. **Use Postgres + PgVector** instead of file-based databases. SQLite/LanceDB/Ladybug don't handle concurrent writes well in containerized environments.

2. **Set `CORS_ALLOWED_ORIGINS`** to your actual frontend domain instead of `*`.

3. **Enable authentication for multi-tenant deployments**: Set `ENABLE_BACKEND_ACCESS_CONTROL=true` (default) and configure user management. For a single-user internal deployment with auth off, set `ENABLE_BACKEND_ACCESS_CONTROL=false`; `REQUIRE_AUTHENTICATION=false` alone is not sufficient when multi-tenant mode is on.

4. **Configure rate limiting**: Set `LLM_RATE_LIMIT_ENABLED=true` to avoid hitting provider limits.

5. **Trace**: Enable OpenTelemetry tracing with `COGNEE_TRACING_ENABLED=true` and an OTLP endpoint. Install with `pip install cognee[tracing]`.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | — | API key for your LLM provider |
| `LLM_MODEL` | No | `openai/gpt-5-mini` | Model identifier |
| `LLM_PROVIDER` | No | `openai` | LLM provider name |
| `DB_PROVIDER` | No | `sqlite` | `sqlite` or `postgres` |
| `DB_HOST` | If postgres | — | Database host |
| `DB_PORT` | If postgres | `5432` | Database port |
| `DB_USERNAME` | If postgres | — | Database user |
| `DB_PASSWORD` | If postgres | — | Database password |
| `DB_NAME` | If postgres | — | Database name |
| `VECTOR_DB_PROVIDER` | No | `lancedb` | `lancedb`, `pgvector`, `chromadb` |
| `GRAPH_DATABASE_PROVIDER` | No | `ladybug` | `ladybug`, `neo4j` |
| `CORS_ALLOWED_ORIGINS` | No | `*` | Allowed CORS origins |
| `ENABLE_BACKEND_ACCESS_CONTROL` | No | `true` | Multi-tenant isolation; when `true`, auth is required |
| `REQUIRE_AUTHENTICATION` | No | inherits from `ENABLE_BACKEND_ACCESS_CONTROL` | Explicit auth override (`false` ignored when multi-tenant is on) |
