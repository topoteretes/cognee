# Deploying Cognee via Coolify

This guide walks you through deploying [Cognee](https://github.com/topoteretes/cognee) to your own self-hosted server using [Coolify](https://coolify.io). 

Coolify is an open-source, self-hosted alternative to platforms like Heroku, Vercel, and Netlify. It simplifies container management, automates SSL certificate provisioning via Let's Encrypt, and manages environment variables and databases from an intuitive web UI.

---

## Prerequisites

Before starting, ensure you have:
1. **A Coolify Instance**: A running instance of Coolify on a virtual private server (VPS). If you don't have one, you can install Coolify on any Ubuntu/Debian server using:
   ```bash
   curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash
   ```
2. **A Domain or Subdomain**: Pointed to your Coolify server's IP address (e.g., `cognee.example.com`).
3. **An LLM API Key**: An API key from a provider like OpenAI (`OPENAI_API_KEY`) or Anthropic (`ANTHROPIC_API_KEY`).
4. **Git Repository (Optional)**: If you plan to deploy custom application logic built on top of Cognee.

---

## Deployment Strategies

You can deploy Cognee onto Coolify using one of two strategies:

| Strategy | Description | Best For | Complexity |
| :--- | :--- | :--- | :--- |
| **Option A: Standalone Container** | Deploys the pre-built `cognee/cognee:latest` image. Uses file-based SQLite, LanceDB, and KuzuDB. | Quick setups, testing, and light workloads | Easiest |
| **Option B: Docker Compose** | Deploys Cognee alongside PostgreSQL (`pgvector`) for structured/vector data storage. | Production environments, high load, and persistent multi-user setups | Moderate |

---

## Option A: Standalone Container (SQLite/LanceDB/KuzuDB)

This option deploys the official Cognee Docker image. All data is stored in file-based databases inside persistent volumes mounted to your host.

### Step 1: Create a New Resource in Coolify
1. Open your Coolify dashboard and navigate to **Projects**.
2. Select your project and environment (e.g., `production`).
3. Click **+ Add New Resource** -> Select **Application**.
4. Choose **Public Docker Image**.
5. Set the Image Name to `cognee/cognee:latest`.
6. Click **Save**.

### Step 2: Configure Routing and Ports
1. In the **General** configuration page, find the **Domains** input.
2. Enter your domain or subdomain (e.g., `https://cognee.example.com`). Coolify will automatically obtain and renew a free Let's Encrypt SSL certificate.
3. Under **Port Mapping**, verify that the **Port** is set to `8000` (this is the default port Cognee listens on inside the container).

### Step 3: Set Up Persistent Volumes
> [!IMPORTANT]
> If you do not configure persistent volumes, your database files, vector indexes, and ingested documents will be lost every time you redeploy or restart the container.

In your application settings, go to the **Storage** tab and add the following volume mounts:

| Volume Name | Mount Path inside Container | Description |
| :--- | :--- | :--- |
| `cognee-data` | `/app/.data_storage` | Stores raw uploaded files and processed document chunks. |
| `cognee-system` | `/app/.cognee_system` | Stores local SQLite database files, LanceDB tables, and Kuzu graph data. |
| `cognee-cache` | `/app/.cognee_cache` | Stores temporary cache files. |
| `cognee-logs` | `/root/.cognee/logs` | Stores application runtime logs. |

### Step 4: Configure Environment Variables
Navigate to the **Environment Variables** tab and add the variables listed in the [Environment Variables Reference](#environment-variables-reference) section below.

---

## Option B: Docker Compose (Cognee + PostgreSQL/pgvector)

For production environments, it is recommended to deploy Cognee using a Docker Compose file that provisions a dedicated PostgreSQL database container with the `pgvector` extension.

### Step 1: Create a Docker Compose Resource
1. Open your Coolify dashboard and navigate to your Project/Environment.
2. Click **+ Add New Resource** -> Select **Application**.
3. Choose **Docker Compose**.
4. Paste the following configuration into the Compose configuration editor:

```yaml
version: '3.8'

services:
  cognee:
    image: cognee/cognee:latest
    restart: always
    environment:
      - DEBUG=false
      - HOST=0.0.0.0
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - REQUIRE_AUTHENTICATION=true
      - FASTAPI_USERS_JWT_SECRET=${FASTAPI_USERS_JWT_SECRET}
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - LLM_MODEL=${LLM_MODEL:-openai/gpt-4o-mini}
      - EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER:-openai}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-openai/text-embedding-3-large}
      - EMBEDDING_DIMENSIONS=${EMBEDDING_DIMENSIONS:-3072}
      # Database Overrides
      - DB_PROVIDER=postgres
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_USERNAME=cognee
      - DB_PASSWORD=${POSTGRES_PASSWORD}
      - DB_NAME=cognee_db
      - VECTOR_DB_PROVIDER=pgvector
    volumes:
      - cognee-data:/app/.data_storage
      - cognee-system:/app/.cognee_system
      - cognee-cache:/app/.cognee_cache
    depends_on:
      postgres:
        condition: service_healthy
    # Coolify uses label rules to assign domains to services
    labels:
      - "coolify.managed=true"

  postgres:
    image: pgvector/pgvector:pg17
    restart: always
    environment:
      POSTGRES_USER: cognee
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: cognee_db
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U cognee -d cognee_db"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  cognee-data:
  cognee-system:
  cognee-cache:
  postgres-data:
```

### Step 2: Configure Environment Variables in Coolify
Coolify will detect the placeholders (`${LLM_API_KEY}`, `${POSTGRES_PASSWORD}`, etc.) and ask you to define them in the environment settings. 

1. **`LLM_API_KEY`**: Your OpenAI or Anthropic API key.
2. **`POSTGRES_PASSWORD`**: A secure, randomly generated string.
3. **`FASTAPI_USERS_JWT_SECRET`**: A secure, randomly generated key for signing auth tokens.

### Step 3: Assign Domain
Under the **General** settings of the `cognee` service, input your target domain (e.g., `https://cognee.example.com`). Coolify handles the internal reverse proxy automatically.

---

## Environment Variables Reference

Configure these environment variables inside your Coolify configuration panel to customize Cognee:

### Core Configurations
| Key | Example Value | Description |
| :--- | :--- | :--- |
| `LLM_API_KEY` | `sk-proj-...` | The API key for your chosen LLM provider. |
| `LLM_PROVIDER` | `openai` | LLM provider name (`openai`, `anthropic`, `cohere`, etc.). |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Model used for classification and structured output. |
| `EMBEDDING_PROVIDER` | `openai` | Embedding vector provider. |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-large` | Embedding model identifier. |
| `EMBEDDING_DIMENSIONS` | `3072` | Number of dimensions for vector generation. |
| `REQUIRE_AUTHENTICATION` | `true` | Set to `true` in production to secure endpoints with API key/JWT login. |
| `FASTAPI_USERS_JWT_SECRET` | `generate-a-secure-random-key` | Secret key for issuing JSON Web Tokens. |

### Database Configurations (Only needed for Option B)
| Key | Value | Description |
| :--- | :--- | :--- |
| `DB_PROVIDER` | `postgres` | Sets PostgreSQL as the relational DB metadata provider. |
| `DB_HOST` | `postgres` | Hostname of the Postgres container (maps to service name). |
| `DB_PORT` | `5432` | Relational database port. |
| `DB_USERNAME` | `cognee` | PostgreSQL login username. |
| `DB_PASSWORD` | `secure-db-password` | Password for PostgreSQL database. |
| `DB_NAME` | `cognee_db` | Relational database name. |
| `VECTOR_DB_PROVIDER` | `pgvector` | Instructs Cognee to utilize PostgreSQL's pgvector extension. |

---

## CI/CD Workflow: Deploying via GitHub Actions

You can automate deployments so that every push to the `main` branch of your repository triggers a build and deployment in Coolify.

### Step 1: Retrieve Coolify Webhook
1. Go to your application in Coolify.
2. Under **Webhooks**, locate the **Deploy Webhook** URL (it looks like `https://coolify.yourdomain.com/api/v1/deploy?uuid=xxxx...`).

### Step 2: Configure GitHub Secrets
1. Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions**.
2. Add a new repository secret:
   - Name: `COOLIFY_WEBHOOK_URL`
   - Value: The webhook URL copied from your Coolify dashboard.

### Step 3: Create GitHub Actions Workflow
Create a new file in your repository at `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Coolify

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Coolify Deployment
        run: |
          curl -X POST -H "Authorization: Bearer ${{ secrets.COOLIFY_WEBHOOK_URL }}"
```

---

## Cost Estimates

Running Cognee on a self-hosted VPS is highly cost-efficient compared to commercial SaaS alternatives.

| Configuration | Recommended Specs | Estimated Cost | Providers |
| :--- | :--- | :--- | :--- |
| **Developer Sandbox** (Option A) | 1 vCPU, 2GB RAM, 20GB SSD | **$4 - $6 / month** | Hetzner Cloud (CX22), DigitalOcean (Basic Droplet) |
| **Production Scale** (Option B) | 2 vCPUs, 4GB-8GB RAM, 50GB SSD | **$10 - $18 / month** | Hetzner Cloud (CPX21/CPX31), AWS Lightsail, Linode |

---

## Troubleshooting

### 1. Database Migrations or Startup Failures
* **Symptom**: Cognee container loops during startup showing database locks or initialization errors.
* **Solution**:
  - In Option A, check that your volume folders are writable by the container. If you run into permission issues on Linux, ensure the host paths mapped to Docker have correct read/write permissions (`chmod -R 777` can test if it is a permission issue, though locking it down to UID `1000` is safer in production).
  - In Option B, ensure the database container is fully healthy before Cognee attempts to start. The `depends_on` rule with a database `healthcheck` is configured in our Compose example to resolve this.

### 2. "Authentication failed" or 401 Unauthorized API access
* **Symptom**: You deploy successfully, but all API requests return `401 Unauthorized`.
* **Solution**:
  - Ensure you have correctly configured the `REQUIRE_AUTHENTICATION` environment variable. 
  - If set to `true`, you must first run a register/login request or issue an API key via your database to connect to Cognee. 
  - For simple, single-user developer instances protected by basic network authorization (like basic auth in Coolify's proxy settings), you can set `REQUIRE_AUTHENTICATION=false` and `ENABLE_BACKEND_ACCESS_CONTROL=false` to bypass backend user authentication.

### 3. LLM Request Timeouts or Failures
* **Symptom**: Ingestion works, but generating vector embeddings or extracting ontology nodes fails.
* **Solution**:
  - Check that the `LLM_API_KEY` is input without quotes or trailing spaces in the Coolify environment variables page.
  - Verify that the server has external internet connectivity to reach the provider APIs (`api.openai.com` or `api.anthropic.com`).
