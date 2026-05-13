"""
Cognee on Modal — 1-click serverless deployment.

Deploys the Cognee FastAPI server as a Modal ASGI app with:
- Persistent volume for file-based databases (SQLite, LanceDB, Ladybug)
- Secret injection for LLM_API_KEY and other credentials
- Auto-scaling with configurable concurrency

Setup:
    1. pip install modal && modal setup
    2. modal secret create cognee-secrets \
         LLM_API_KEY=sk-xxx \
         LLM_MODEL=openai/gpt-4o-mini
    3. modal deploy distributed/deploy/modal_app.py

Your API will be available at:
    https://<your-org>--cognee-api-serve.modal.run
"""

import modal

app = modal.App("cognee-api")

# Persistent volume for file-based databases
volume = modal.Volume.from_name("cognee-data", create_if_missing=True)

# Build image from existing Dockerfile
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "libpq-dev", "git", "curl", "cmake", "clang", "build-essential")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system cognee[postgres,api]",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HOST": "0.0.0.0",
            "DATA_ROOT_DIRECTORY": "/data/cognee_data",
            "SYSTEM_ROOT_DIRECTORY": "/data/cognee_system",
        }
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("cognee-secrets")],
    volumes={"/data": volume},
    timeout=3600,
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def serve():
    from cognee.api.client import app as fastapi_app

    return fastapi_app
