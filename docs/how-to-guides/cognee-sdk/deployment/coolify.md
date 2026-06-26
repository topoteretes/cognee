---
title: "Deploy Cognee on Coolify"
description: "A step-by-step guide to deploying a production-ready Cognee instance using Coolify."
---

# Deploy Cognee on Coolify

## Introduction

[Coolify](https://coolify.io/) is an open-source, self-hosted Heroku/Netlify alternative. It simplifies the deployment of applications, databases, and services directly to your own servers. 

Using Coolify with Cognee allows you to effortlessly host your own private AI memory platform. By using this guide, you will deploy the core Cognee API server, the MCP (Model Context Protocol) server, and the necessary databases using Coolify's native Docker Compose integration.

## Prerequisites

Before starting, ensure you have the following:
- A running **Coolify instance** (version 4.x recommended).
- A **GitHub account** connected to your Coolify instance.
- An **OpenAI API Key** (or another supported LLM provider key).
- Basic familiarity with Git and environment variables.
- Docker installed on the server hosting Coolify.

## Step 1 — Prepare the Project

Coolify pulls your application directly from a Git repository. To deploy Cognee:

1. Log into your GitHub account.
2. Fork the official [Cognee repository](https://github.com/topoteretes/cognee) to your own account.
3. In Coolify, navigate to your server and click **Add New Resource**.
4. Select **Git Repository** (or **GitHub App** if you have integrated it).
5. Choose your forked `cognee` repository and the `main` branch.

## Step 2 — Configure Environment Variables

Cognee requires certain environment variables to function correctly. Coolify allows you to securely inject these into your containers.

In the Coolify project dashboard for your new resource, navigate to the **Environment Variables** tab and add the following required variables:

| Variable | Purpose | Example Value | Required |
|----------|---------|---------------|----------|
| `LLM_API_KEY` | Your language model API key | `sk-...` | Yes |
| `LLM_PROVIDER` | The LLM provider to use | `openai` | No (defaults to `openai`) |
| `ENVIRONMENT` | Deployment environment | `production` | No (defaults to `local`) |

### Example `.env`

You can click the **Bulk Edit** button in Coolify and paste the following:

```env
LLM_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
ENVIRONMENT=production
LOG_LEVEL=INFO
```

> [!TIP]
> For advanced configurations (like changing embedding models or adjusting rate limits), refer to the `.env.template` file in the root of the repository.

## Step 3 — Database Setup

Cognee supports multiple databases. By default, it uses file-based databases (`sqlite`, `lancedb`, `kuzu`) which require zero setup and are perfect for getting started. 

If you are using the default file-based setup, you can skip to Step 4. Coolify will automatically mount volumes to persist your data.

### Using External Databases (Optional)

If you prefer using Postgres, Neo4j, or Redis, you can provision them directly inside Coolify:
1. In your Coolify project, click **Add New Resource** -> **Databases**.
2. Provision your preferred database (e.g., PostgreSQL).
3. Once provisioned, note the internal URL (e.g., `postgresql://user:pass@postgres:5432/cognee_db`).
4. Add the connection strings to your Cognee environment variables:
   - `DB_PROVIDER=postgres`
   - `DB_HOST=postgres`
   - `DB_PORT=5432`
   - `DB_USERNAME=user`
   - `DB_PASSWORD=pass`
   - `DB_NAME=cognee_db`

## Step 4 — Deploy with Coolify

With the repository connected and environment variables set, configure the deployment:

1. **Build Pack**: Ensure Coolify detects the project as **Docker Compose**.
2. **Compose File**: Set the Compose file path to `/docker-compose.yml`.
3. **Domains**: In the **Configuration** -> **General** tab, set your custom domain for the Cognee API (e.g., `https://cognee.yourdomain.com`).
4. **Ports**: Ensure the internal port matches the Cognee API port (`8000`).
5. **SSL**: Coolify will automatically provision a Let's Encrypt SSL certificate if you specify an `https://` domain.
6. **Deploy**: Click the **Deploy** button. Coolify will begin building the images and orchestrating the containers.

## Step 5 — Verify Deployment

Once the deployment completes, verify that Cognee is running:

1. Navigate to the **Logs** tab in Coolify to ensure there are no startup errors.
2. Open your browser and visit your domain's health check endpoint: 
   `https://cognee.yourdomain.com/health`
3. You should see a successful response indicating the server is healthy.

### Example Terminal Verification

You can also verify the API from your terminal:

```bash
curl -X GET https://cognee.yourdomain.com/health
```

## Troubleshooting

### Failed Docker Builds
- **Symptoms**: Deployment hangs or fails during the `build` phase.
- **Cause**: Outdated Docker engine on the host or insufficient server RAM.
- **Solution**: Ensure your server has at least 4GB of RAM (8GB recommended). Check Coolify server metrics.

### Environment Variable Mistakes
- **Symptoms**: Cognee starts but throws authentication or connection errors in the logs.
- **Cause**: Typo in the `LLM_API_KEY` or missing required variables.
- **Solution**: Verify the variables in the **Environment Variables** tab. Remember to click "Save" and restart the container.

### Database Connection Failures
- **Symptoms**: `Connection refused` or `Timeout` errors in the logs.
- **Cause**: Incorrect `DB_HOST` or the database container is not running on the same Coolify network.
- **Solution**: Ensure the database and Cognee are deployed in the same Coolify project/environment, so they share the internal Docker network.

### Volume Permissions
- **Symptoms**: `Permission denied` errors when creating SQLite files.
- **Cause**: Docker container lacks write access to the mounted volume.
- **Solution**: Verify Coolify's volume mount configurations. Coolify usually handles this automatically, but you may need to manually inspect the host directory permissions.

### Startup Failures
- **Symptoms**: Container crashes loop with exit codes.
- **Cause**: Migration failure or missing ports configuration.
- **Solution**: Ensure the `8000` port is exposed correctly. Check application logs for detailed python stack traces.

## Cost Estimate

Running Cognee requires adequate memory for the vector and graph operations. Here is a rough estimate for hosting on common VPS providers (e.g., Hetzner, DigitalOcean, AWS EC2):

- **Small Production** (4 vCPUs, 8GB RAM): ~$10 - $40 / month
- **Scaling / Large Graph** (8 vCPUs, 16GB RAM): ~$20 - $80 / month

> [!NOTE]
> This estimate does not include the cost of LLM API calls (e.g., OpenAI tokens).

## Best Practices

- **Backups**: If using file-based databases, configure Coolify's built-in backup solutions to backup the Docker volumes.
- **Secrets Management**: Never commit your `.env` file or API keys to GitHub. Always use Coolify's secure environment variables manager.
- **HTTPS**: Always use Coolify's auto-SSL feature to encrypt data in transit.
- **Monitoring**: Utilize Coolify's dashboard to monitor CPU and Memory usage. Cognee's vector operations can be memory-intensive.

## Docker Compose Example

For reference, Coolify will parse the `docker-compose.yml` included in the root of the repository. It includes the following key services:

- **`cognee`**: The main API server running on port `8000`.
- **`cognee-mcp`**: The Model Context Protocol server (useful for IDE integrations).
- **`frontend`**: An optional UI for managing your knowledge graph.
- **Optional Databases**: Commented-out profiles for Postgres, Neo4j, and Redis.
