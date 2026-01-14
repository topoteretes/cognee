# Deploy REST API Server

> Deploy Cognee as a REST API server using Docker or Python

Deploy Cognee as a REST API server to expose its functionality via HTTP endpoints.

## Setup

```bash  theme={null}
# Clone repository
git clone https://github.com/topoteretes/cognee.git
cd cognee

# Configure environment
cp .env.template .env
```

<Info>
  Edit `.env` with your preferred configuration. See [Setup Configuration](/setup-configuration/overview) guides for all available options.
</Info>

## Deployment Methods

<Tabs>
  <Tab title="Docker">
    ### Start Server

    ```bash  theme={null}
    # Start API server
    docker compose up --build cognee

    # Check status
    docker compose ps
    ```
  </Tab>

  <Tab title="Python (Local)">
    ### Setup

    ```bash  theme={null}
    # Create virtual environment
    uv venv && source .venv/bin/activate

    # Install with all extras
    uv sync --all-extras
    ```

    ### Start Server

    ```bash  theme={null}
    # Run API server
    uvicorn cognee.api.client:app --host 0.0.0.0 --port 8000
    ```
  </Tab>
</Tabs>

## Access API

* **API:** [http://localhost:8000](http://localhost:8000)
* **Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

## Authentication

If `REQUIRE_AUTHENTICATION=true` in your `.env` file:

1. **Register:** `POST /api/v1/auth/register`
2. **Login:** `POST /api/v1/auth/login`
3. **Use token:** Include `Authorization: Bearer <token>` header or use cookies

## API Examples

<AccordionGroup>
  <Accordion title="Authentication">
    **Register a user:**

    ```bash  theme={null}
    curl -X POST "http://localhost:8000/api/v1/auth/register" \
      -H "Content-Type: application/json" \
      -d '{"email": "user1@example.com", "password": "strong_password"}'
    ```

    **Login and get token:**

    ```bash  theme={null}
    TOKEN="$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
      -H 'Content-Type: application/x-www-form-urlencoded' \
      -d 'username=user1@example.com&password=strong_password' | jq -r .access_token)"
    ```
  </Accordion>

  <Accordion title="Dataset Management">
    **Create a dataset:**

    ```bash  theme={null}
    curl -X POST http://localhost:8000/api/v1/datasets \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{"name": "project_docs"}'
    ```

    **List datasets:**

    ```bash  theme={null}
    curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/datasets
    ```
  </Accordion>

  <Accordion title="Data Operations">
    **Add data (upload file):**

    ```bash  theme={null}
    curl -X POST http://localhost:8000/api/v1/add \
      -H "Authorization: Bearer $TOKEN" \
      -F "data=@/absolute/path/to/file.pdf" \
      -F "datasetName=project_docs"
    ```

    **Build knowledge graph:**

    ```bash  theme={null}
    curl -X POST http://localhost:8000/api/v1/cognify \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{"datasets": ["project_docs"]}'
    ```

    **Search data:**

    ```bash  theme={null}
    curl -X POST http://localhost:8000/api/v1/search \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{"query": "What are the main topics?", "datasets": ["project_docs"], "top_k": 10}'
    ```
  </Accordion>

  <Accordion title="Multi-tenant Operations">
    **Create tenant:**

    ```bash  theme={null}
    curl -X POST "http://localhost:8000/api/v1/permissions/tenants?tenant_name=acme" \
      -H "Authorization: Bearer $TOKEN"
    ```

    **Add user to tenant:**

    ```bash  theme={null}
    curl -X POST "http://localhost:8000/api/v1/permissions/users/<user_id>/tenants?tenant_id=<tenant_id>" \
      -H "Authorization: Bearer $TOKEN"
    ```

    **Create role:**

    ```bash  theme={null}
    curl -X POST "http://localhost:8000/api/v1/permissions/roles?role_name=editor" \
      -H "Authorization: Bearer $TOKEN"
    ```

    **Assign user to role:**

    ```bash  theme={null}
    curl -X POST "http://localhost:8000/api/v1/permissions/users/<user_id>/roles?role_id=<role_id>" \
      -H "Authorization: Bearer $TOKEN"
    ```

    **Grant dataset permissions:**

    ```bash  theme={null}
    curl -X POST "http://localhost:8000/api/v1/permissions/datasets/<principal_id>?permission_name=read&dataset_ids=<ds_uuid_1>&dataset_ids=<ds_uuid_2>" \
      -H "Authorization: Bearer $TOKEN"
    ```
  </Accordion>
</AccordionGroup>

<Columns cols={3}>
  <Card title="API Reference" icon="book" href="/api-reference/introduction">
    Explore all API endpoints
  </Card>

  <Card title="Setup Configuration" icon="settings" href="/setup-configuration/overview">
    Configure providers and databases
  </Card>

  <Card title="MCP Integration" icon="plug" href="/cognee-mcp/mcp-overview">
    Set up AI assistant integration
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt