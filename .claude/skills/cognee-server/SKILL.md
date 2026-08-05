---
name: cognee-server
description: Use when the user wants to run the cognee API server (and optional UI) on their own machine — starting it, checking it's healthy, connecting the SDK or other clients to it, and choosing the right auth posture.
---

# Start the cognee server locally

## From an installed cognee (no Docker)

```bash
cognee-cli -ui
```

launches the full local stack: FastAPI backend on **http://localhost:8000**
and the UI on **http://localhost:3000**. Needs `LLM_API_KEY` in the
environment or `.env`. The interactive API reference is at
http://localhost:8000/docs, health at `/health`.

For API-only serving via Docker instead, use the cognee-docker skill (the
prebuilt `cognee/cognee:main` image or `docker compose up` from the repo).

## Auth posture

`ENABLE_BACKEND_ACCESS_CONTROL` decides everything:

- `true` (default): multi-tenant — auth required on every API call, per
  user+dataset database isolation.
- `false`: single-user local mode — no auth, shared local databases. Right
  choice for a personal dev server; never for anything exposed.

`REQUIRE_AUTHENTICATION=false` is ignored while access control is on; to turn
auth off you must set `ENABLE_BACKEND_ACCESS_CONTROL=false`.

## Connecting clients to the running server

- **SDK / CLI against the server** (instead of embedded local mode):

  ```bash
  cognee-cli serve --url http://localhost:8000        # local instance
  cognee-cli serve                                    # cognee cloud (device flow)
  cognee-cli serve --logout                           # disconnect
  ```

  In Python: `await cognee.serve(url="http://localhost:8000")`.

- **HTTP**: main routes live under `/api/v1/` — `add`, `cognify`, `search`,
  `memify`, `datasets`, `users`, `visualize` (see `cognee/api/v1/`).

## Graph visualization without the full UI

```python
from cognee.api.v1.visualize import visualization_server

shutdown = visualization_server(port=8080)  # synchronous; returns a shutdown callable
```

## Troubleshooting

- Port 8000 already taken → stop the other service or remap (compose:
  `"8080:8000"`).
- 401/403 on every call → you're in multi-tenant mode; either authenticate or
  set `ENABLE_BACKEND_ACCESS_CONTROL=false` and restart.
- Search returns `[]` instead of erroring → permission-filtered result;
  check dataset access rights for the calling user.
