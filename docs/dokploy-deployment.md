---
title: "Dokploy Deployment"
description: "Deploy Cognee on your own server using Dokploy, a self-hosted PaaS built on Docker and Traefik"
---

# Deploying Cognee with Dokploy

[Dokploy](https://dokploy.com) is a free, open-source Platform-as-a-Service (PaaS) built on
Docker and Traefik. It gives you a web dashboard to deploy and manage containerised applications
on any VPS — no complex DevOps knowledge required.

This guide walks you through deploying a Cognee sample app on Dokploy: from a fresh server to a
running, publicly accessible API endpoint with optional database and session storage.

---

## What Is Dokploy?

Dokploy wraps Docker and Traefik behind a clean browser-based dashboard. You connect your server
once, and from then on you deploy containers, manage environment variables, configure domains, and
monitor logs — all from the UI. It supports Docker Compose natively, which makes it a natural fit
for Cognee's multi-service architecture.

Key features relevant to this guide:

- One-line installer that sets up Docker Swarm, Traefik, and Dokploy itself
- Docker Compose service support with environment variable management
- Built-in managed databases (PostgreSQL, Redis, and more)
- Automatic SSL certificates via Let's Encrypt through Traefik
- Real-time log streaming and health monitoring

---

## What You Will Deploy

By the end of this guide you will have:

- The **Cognee API server** running on port `8000` (using the prebuilt `cognee/cognee:main` image)
- Persistent storage volumes for Cognee's data and system directories
- A working **health check** endpoint to confirm the service is live
- Optional: **PostgreSQL** for production-grade relational storage
- Optional: **Session caching** with Redis or filesystem backend
- Optional: **MCP server** alongside the API

---

## Prerequisites

| Requirement | Details |
|---|---|
| A VPS or dedicated server | Ubuntu 22.04 LTS recommended; minimum 2 vCPU / 2 GB RAM |
| Root SSH access | The Dokploy installer requires root or sudo privileges |
| An LLM API key | OpenAI, Anthropic, Gemini, or any [supported provider](https://docs.cognee.ai/setup-configuration/llm-providers) |
| A domain name (optional) | Required for HTTPS via Traefik and Let's Encrypt |
| Basic Linux familiarity | Running shell commands and editing files over SSH |

> Dokploy installs Docker and configures Docker Swarm automatically. You do **not** need to
> pre-install Docker.

---

## Step 1: Install Dokploy on Your Server

SSH into your server as root and run the official one-line installer:

```bash
curl -sSL https://dokploy.com/install.sh | sh
```

The script installs Docker, enables Docker Swarm mode, and deploys the Dokploy application stack.
When it finishes you will see output similar to:

```
Congratulations, Dokploy is installed!
Wait 15 seconds for the server to start
Please go to http://YOUR-SERVER-IP:3000
```

**If you use a firewall (UFW), open the required ports before continuing:**

```bash
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP (needed for Let's Encrypt challenge)
ufw allow 443/tcp   # HTTPS
ufw allow 3000/tcp  # Dokploy dashboard
ufw enable
```

Open `http://YOUR-SERVER-IP:3000` in your browser and create your administrator account.

---

## Step 2: Create a Project in Dokploy

1. Log in to the Dokploy dashboard at `http://YOUR-SERVER-IP:3000`.
2. In the left sidebar, click **Projects** → **Create Project**.
3. Name it `cognee` (or any name you prefer) and click **Save**.

---

## Step 3: Add a Docker Compose Service

> **Important — do not copy the repo's `docker-compose.yml` directly.**
>
> The `docker-compose.yml` file in the Cognee repository is optimised for **local development**.
> It uses two bind mounts that do not work in a remote Dokploy deployment:
>
> - `./cognee:/app/cognee` — mounts the local source directory into the container. On a remote
>   server there is no checked-out source tree, so this path does not exist and the container
>   will fail to start.
> - `.env:/app/.env` — mounts a local `.env` file. Dokploy manages environment variables through
>   its own UI and injects them directly; a bind-mounted `.env` file is not needed and will not
>   be present on the remote server.
>
> The compose definition below uses the **prebuilt Docker Hub image** and **named volumes**
> instead — the correct approach for any managed remote environment like Dokploy.

Cognee publishes prebuilt images to Docker Hub on every push to `main`. You do not need to clone
the repository or build anything locally.

Inside your `cognee` project, click **Create Service** → **Docker Compose**.

Paste the following compose definition into the editor:

```yaml
services:
  cognee:
    image: cognee/cognee:main
    ports:
      - "8000:8000"
    volumes:
      - cognee_data:/app/cognee/.data_storage
      - cognee_system:/app/cognee/.cognee_system
    restart: unless-stopped

volumes:
  cognee_data:
  cognee_system:
```

Key differences from the repo's local compose file:

- Uses `image: cognee/cognee:main` (prebuilt from Docker Hub) instead of `build: .`
- No `./cognee:/app/cognee` source bind mount — not available on a remote server
- No `.env:/app/.env` bind mount — Dokploy injects environment variables directly (see Step 4)
- Named volumes (`cognee_data`, `cognee_system`) replace any local paths — these persist across
  redeployments on the remote server

> **Why are named volumes required?** Even when Cognee is pointed at external databases, it still
> writes ingestion artefacts, file caches, and loader outputs to `DATA_ROOT_DIRECTORY`
> (default `.data_storage`) and `SYSTEM_ROOT_DIRECTORY` (default `.cognee_system`). Without
> persistent named volumes, all ingested data is lost on every redeploy.

---

## Step 4: Add Environment Variables

In the Dokploy service editor, open the **Environment** tab and add your configuration.

> **Important — do not wrap values in quotes.** Dokploy passes environment variables to the
> container literally, including any surrounding quotes (the same way a Docker `--env-file`
> does — quotes are **not** stripped, unlike a shell). A value such as `LITELLM_LOG="ERROR"`
> is delivered as the literal string `"ERROR"` (quotes included), which crashes Cognee on
> startup with `AttributeError: module 'logging' has no attribute '"ERROR"'`. Always write
> `LITELLM_LOG=ERROR`, never `LITELLM_LOG="ERROR"`.

### Minimum required setup

```env
LLM_API_KEY=your_openai_api_key
```

That single variable is all that is needed to start. Everything else uses sensible defaults
(SQLite for relational storage, LanceDB for vector storage, Ladybug for graph storage — all
file-based, no extra setup required).

### Recommended production additions

```env
# LLM provider
LLM_API_KEY=your_openai_api_key
LLM_MODEL=openai/gpt-5-mini
LLM_PROVIDER=openai

# Restrict CORS in production
CORS_ALLOWED_ORIGINS=https://yourdomain.com

# Skip LLM connectivity check on startup (useful if services start in parallel)
COGNEE_SKIP_CONNECTION_TEST=false

# Logging
LITELLM_LOG=ERROR
ENV=local
```

> **A note on authentication.** By default `ENABLE_BACKEND_ACCESS_CONTROL=true`, which turns
> multi-tenant mode **on** and therefore requires authentication (JWT) on all API calls —
> `REQUIRE_AUTHENTICATION` inherits that value when it is unset. Setting
> `REQUIRE_AUTHENTICATION=false` on its own is **ignored** while access control is on (Cognee
> logs a warning and forces auth back on). For a genuine single-user, no-auth deployment you
> must disable access control explicitly:
>
> ```env
> ENABLE_BACKEND_ACCESS_CONTROL=false
> REQUIRE_AUTHENTICATION=false
> ```
>
> Only do this for a private, single-user instance — never on a publicly reachable endpoint.

### Full environment variable reference

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | *(required)* | API key for your LLM provider |
| `LLM_MODEL` | `openai/gpt-5-mini` | LLM model identifier |
| `LLM_PROVIDER` | `openai` | Provider: `openai`, `anthropic`, `gemini`, `ollama`, … |
| `LLM_ENDPOINT` | — | Custom LLM API endpoint (for Azure, local, or custom providers) |
| `EMBEDDING_PROVIDER` | `openai` | Embedding provider (defaults to `LLM_PROVIDER` if unset) |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-large` | Embedding model |
| `DB_PROVIDER` | `sqlite` | Relational DB: `sqlite` or `postgres` |
| `DB_HOST` | — | Postgres host (when `DB_PROVIDER=postgres`) |
| `DB_PORT` | `5432` | Postgres port |
| `DB_USERNAME` | `cognee` | Postgres username |
| `DB_PASSWORD` | — | Postgres password |
| `DB_NAME` | `cognee_db` | Postgres database name |
| `GRAPH_DATABASE_PROVIDER` | `ladybug` | Graph DB: `ladybug`, `kuzu`, `neo4j`, `postgres`, … |
| `VECTOR_DB_PROVIDER` | `lancedb` | Vector DB: `lancedb`, `pgvector`, `chromadb`, … |
| `DATA_ROOT_DIRECTORY` | `.data_storage` | Where Cognee stores raw data files |
| `SYSTEM_ROOT_DIRECTORY` | `.cognee_system` | Where Cognee stores database files |
| `CORS_ALLOWED_ORIGINS` | `*` | Restrict to your domain in production |
| `REQUIRE_AUTHENTICATION` | *(inherits `ENABLE_BACKEND_ACCESS_CONTROL`)* | Require JWT auth on all API calls. Unset → follows access control (so effectively `true` by default). Ignored if set `false` while access control is `true`. |
| `ENABLE_BACKEND_ACCESS_CONTROL` | `true` | Multi-tenant isolation per user and dataset |
| `FASTAPI_USERS_JWT_SECRET` | `super_secret` | **Change this** to a long random string in production |
| `JWT_LIFETIME_SECONDS` | `3600` | Token lifetime in seconds |
| `COGNEE_SKIP_CONNECTION_TEST` | `false` | Skip LLM connectivity check at startup |
| `CACHING` | `true` | Session caching is on by default (set `false` to disable) |
| `CACHE_BACKEND` | `sqlite` | Session cache backend: `sqlite`, `fs`, `redis`, or `postgres` |
| `CACHE_HOST` | `localhost` | Redis hostname (when `CACHE_BACKEND=redis`) |
| `CACHE_PORT` | `6379` | Redis port |
| `LLM_RATE_LIMIT_ENABLED` | `false` | Enable client-side rate limiting for LLM calls |
| `LLM_RATE_LIMIT_REQUESTS` | `60` | Max requests per interval |
| `LLM_RATE_LIMIT_INTERVAL` | `60` | Interval in seconds |
| `LITELLM_LOG` | `ERROR` | LLM log verbosity: `ERROR`, `INFO`, `DEBUG` |
| `TELEMETRY_DISABLED` | `0` | Set `1` to disable telemetry |

See the full `.env.template` in the repository for every available option:
[`.env.template`](https://github.com/topoteretes/cognee/blob/main/.env.template)

---

## Step 5: Set Up Session Storage (Optional)

Sessions give Cognee short-term conversational memory across search calls. A session is
identified by `(user_id, session_id)` and stores recent queries, responses, and context.
Sessions are used automatically when you pass a `session_id` to `cognee.search()` or the
`/api/v1/search` endpoint.

Session caching is **on by default** (`CACHING=true`, backend `sqlite`), so sessions work with no
extra configuration. You only need these variables to change the backend. As in Step 4, enter them
**without surrounding quotes**.

To use the filesystem backend instead of SQLite:

```env
CACHING=true
CACHE_BACKEND=fs         # filesystem cache, no extra service needed
```

For multi-process or distributed setups, use Redis instead:

```env
CACHING=true
CACHE_BACKEND=redis
CACHE_HOST=your-redis-host
CACHE_PORT=6379
```

> The filesystem backend stores sessions in `{DATA_ROOT_DIRECTORY}/.cognee_fs_cache/sessions_db`.
> Redis is recommended for production because it supports shared locks for concurrent database
> access. The Dokploy managed Redis service (see Step 7) can be used directly.

---

## Step 6: Set Up the Database (Optional)

The default file-based databases (SQLite, LanceDB, Ladybug) work out of the box for single-user or
development use. For production, PostgreSQL with pgvector is the recommended relational and vector
database backend.

### Create a managed PostgreSQL database in Dokploy

1. In the Dokploy sidebar, click **Databases** → **Create Database** → **PostgreSQL**.
2. Set the name to `cognee-db`, choose a password, and click **Create**.
3. Note the internal connection hostname that Dokploy generates (typically something like
   `cognee-db-postgres`).

### Update your environment variables

Remember: enter these in the Dokploy **Environment** tab **without surrounding quotes** (see the
warning in Step 4).

```env
# Relational database
DB_PROVIDER=postgres
DB_HOST=cognee-db-postgres   # internal Dokploy hostname
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=your_password
DB_NAME=cognee_db

# Vector database (pgvector, shares the same Postgres instance)
VECTOR_DB_PROVIDER=pgvector
VECTOR_DB_URL=postgresql://postgres:your_password@cognee-db-postgres:5432/cognee_db

# Graph database (keep the default Ladybug for single-node, switch to neo4j for multi-agent)
GRAPH_DATABASE_PROVIDER=ladybug
```

> **Multi-agent note:** The default Ladybug graph store uses file-based locking and is not suitable
> for concurrent access from multiple agents. Switch to Neo4j or the Postgres graph backend for
> multi-agent deployments.

---

## Step 7: Deploy

Click **Deploy** in the Dokploy service panel. Dokploy pulls the `cognee/cognee:main` image and
starts the container. Watch the real-time log output in the **Logs** tab.

The image runs under **Gunicorn with Uvicorn workers**. A successful startup produces output like
this:

```
Running database migrations...
Database migrations done.
Starting server...
[INFO] Starting gunicorn 23.0.0
[INFO] Listening at: http://0.0.0.0:8000 (1)
[INFO] Using worker: uvicorn.workers.UvicornWorker
[INFO] Application startup complete.
```

If the startup fails, check the Logs tab first. The most common cause is a missing or incorrect
`LLM_API_KEY`.

---

## Step 8: Test the API and Health Check

Once the service is running, verify it is working correctly.

### Check the health endpoint

```bash
curl http://YOUR-SERVER-IP:8000/health
```

Expected response:

```json
{"status": "ready", "health": "healthy", "version": "1.2.2-local"}
```

### Open the interactive API docs

Navigate to `http://YOUR-SERVER-IP:8000/docs` in your browser. This is Cognee's FastAPI Swagger
UI where you can explore and test every endpoint interactively.

### Run a quick end-to-end test

```bash
# 1. Register a user (when REQUIRE_AUTHENTICATION=false this step is optional)
curl -X POST "http://YOUR-SERVER-IP:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "testpassword123"}'

# 2. Log in and capture the token
TOKEN=$(curl -s -X POST "http://YOUR-SERVER-IP:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=testpassword123" | jq -r .access_token)

# 3. Add a file to the knowledge graph.
#    The `data` field is a file upload, so write the text to a file first and
#    reference it with `@` — sending a bare string returns a 422 validation error.
echo "Cognee is an open-source AI memory platform." > note.txt
curl -X POST "http://YOUR-SERVER-IP:8000/api/v1/add" \
  -H "Authorization: Bearer $TOKEN" \
  -F "data=@note.txt" \
  -F "datasetName=test_dataset"

# 4. Build the knowledge graph from the ingested data
curl -X POST "http://YOUR-SERVER-IP:8000/api/v1/cognify" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"datasets": ["test_dataset"]}'

# 5. Search the knowledge graph
curl -X POST "http://YOUR-SERVER-IP:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "What is Cognee?", "datasets": ["test_dataset"]}'
```

A successful search returns a list of relevant results from the knowledge graph.

---

## Step 9: Configure a Domain and HTTPS (Recommended)

Dokploy uses Traefik to manage routing and SSL certificates automatically through Let's Encrypt.

1. In your DNS provider, add an **A record** pointing your domain to `YOUR-SERVER-IP`.
2. In the Dokploy service settings, open the **Domains** tab.
3. Click **Add Domain**, enter your domain (e.g. `cognee.yourdomain.com`), enable **HTTPS**, and
   save.
4. Dokploy provisions the Let's Encrypt certificate automatically — no manual certificate setup
   required.

Once the certificate is issued, your Cognee API is accessible at:

- API: `https://cognee.yourdomain.com`
- Docs: `https://cognee.yourdomain.com/docs`
- Health: `https://cognee.yourdomain.com/health`

---

## Optional: Enable the Web UI

The Cognee frontend is an experimental web interface.

> **There is no prebuilt `cognee/cognee-frontend` image on Docker Hub.** Only `cognee/cognee`
> (the API) and `cognee/cognee-mcp` (the MCP server) are published. The frontend must therefore
> be **built from source**, which means pointing Dokploy at the Git repository rather than using
> the compose-with-prebuilt-image approach used above for the API.

To add it, create a **second Dokploy service** of type **Docker Compose**, set its build context to
the cloned repo's `cognee-frontend/` directory, and use a compose definition like this:

```yaml
services:
  frontend:
    build:
      context: ./cognee-frontend
      dockerfile: Dockerfile
    ports:
      - "3001:3000"
    environment:
      NEXT_PUBLIC_BACKEND_API_URL: "http://YOUR-SERVER-IP:8000"
    restart: unless-stopped
```

> The frontend runs on port `3001` here to avoid conflicting with the Dokploy dashboard on
> port `3000`. Adjust as needed.
>
> The correct environment variable is `NEXT_PUBLIC_BACKEND_API_URL` (this is what the frontend
> reads — see the repository's `docker-compose.yml` and `cognee-frontend/.env.template`).
>
> Note: do not add `env_file: .env` or source bind mounts to any service in a Dokploy deployment.
> Dokploy injects all environment variables through its UI; `.env` files are a local-only pattern.

---

## Optional: Enable the MCP Server

To run the Cognee MCP server alongside the API (for IDE integrations like Cursor or Claude Code),
add a second service to your compose definition:

```yaml
services:
  cognee:
    image: cognee/cognee:main
    ports:
      - "8000:8000"
    volumes:
      - cognee_data:/app/cognee/.data_storage
      - cognee_system:/app/cognee/.cognee_system
    restart: unless-stopped

  cognee-mcp:
    image: cognee/cognee-mcp:main
    ports:
      - "8001:8000"
    environment:
      TRANSPORT_MODE: "sse"
    restart: unless-stopped

volumes:
  cognee_data:
  cognee_system:
```

> The MCP server's default transport in the repo's `docker-compose.yml` is `sse` (Server-Sent
> Events). The guide uses `sse` here to match that default. Both services share environment
> variables set in the Dokploy UI — no `env_file` bind mount is needed or supported in a remote
> Dokploy deployment.

The MCP server is available at `http://YOUR-SERVER-IP:8001/sse`. Configure your IDE's MCP client
to point to that URL.

---

## Troubleshooting

### Container exits immediately on startup

Check the **Logs** tab in Dokploy. A missing or invalid `LLM_API_KEY` is the most common cause.
Verify the value, update the environment variable in Dokploy, and redeploy.

### `AttributeError: module 'logging' has no attribute '"ERROR"'` (or similar quoted-value error)

You wrapped an environment variable value in quotes in the Dokploy **Environment** tab. Dokploy
passes values literally including the quotes, so `LITELLM_LOG="ERROR"` becomes the string
`"ERROR"` (with quotes) and crashes startup. Remove the surrounding quotes from **all** values in
the Environment tab — write `LITELLM_LOG=ERROR`, not `LITELLM_LOG="ERROR"` — and redeploy.

### `PermissionError: [Errno 13] Permission denied` on `.data_storage`

This means the container cannot write to its data directory. The named volumes in the compose
definition above fix this. If you deployed without volumes, add them and redeploy. Cognee always
requires writable paths for `DATA_ROOT_DIRECTORY` and `SYSTEM_ROOT_DIRECTORY` even when external
databases are configured.

### `[Errno 111] Connection refused` when using PostgreSQL

Cognee started before PostgreSQL finished initialising. Set `COGNEE_SKIP_CONNECTION_TEST=true`
as a temporary workaround while the database comes up, then remove it after the first successful
deployment. Alternatively, use Dokploy's service ordering to ensure the database starts first.

### API returns `403 Forbidden` (or `401 Unauthorized`) on search

When `ENABLE_BACKEND_ACCESS_CONTROL=true` (the default), all API calls require a valid JWT.
For a single-user deployment without auth, set both (remember: no quotes in the Environment tab):

```env
ENABLE_BACKEND_ACCESS_CONTROL=false
REQUIRE_AUTHENTICATION=false
```

### Domain is not resolving after adding it in Dokploy

DNS propagation can take up to 48 hours. Verify the A record is correct with:

```bash
nslookup cognee.yourdomain.com
```

Also check that ports 80 and 443 are open in your firewall (required for the Let's Encrypt
challenge).

### LLM rate limit errors

Enable client-side rate limiting to avoid hitting your provider's limits:

```env
LLM_RATE_LIMIT_ENABLED=true
LLM_RATE_LIMIT_REQUESTS=60
LLM_RATE_LIMIT_INTERVAL=60
```

### Session data is not persisting between searches

Confirm that `CACHING=true` is set and that `CACHE_BACKEND` matches an available backend.
For filesystem caching, the named `cognee_data` volume must be mounted so the cache directory
at `{DATA_ROOT_DIRECTORY}/.cognee_fs_cache/` survives redeployments.

---

## Next Steps

- Browse [other deployment options](https://docs.cognee.ai/how-to-guides/cognee-sdk/deployment) — Modal, EC2,
  Kubernetes, and Render
- Read the [Setup & Configuration guide](https://docs.cognee.ai/setup-configuration/overview) for the full environment
  variable reference
- Explore [LLM provider options](https://docs.cognee.ai/setup-configuration/llm-providers) to switch from OpenAI to
  Anthropic, Gemini, Ollama, or any other supported provider
- Learn about [Sessions and Caching](https://docs.cognee.ai/core-concepts/sessions-and-caching) for conversational
  memory
- Try the [Deploy REST API Server guide](https://docs.cognee.ai/guides/deploy-rest-api-server) for the full HTTP API
  reference with authentication examples
- Join the [Discord community](https://discord.gg/NQPKmU5CCg) for deployment support
