"""FastAPI server for the Cognee API."""

import os
import time
import asyncio
from datetime import datetime, timezone

import uvicorn
import sentry_sdk
from traceback import format_exc
from contextlib import asynccontextmanager
from fastapi import Request
from fastapi import FastAPI, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from sqlalchemy import text

from cognee.exceptions import CogneeApiError
from cognee.shared.logging_utils import get_logger, setup_logging
from cognee.version import get_cognee_version

# Critical deps
from cognee.infrastructure.databases.relational import (
    get_relational_engine,
    get_relational_config,
)
from cognee.infrastructure.databases.vector import (
    get_vector_engine,
)
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph import config as graph_config_module
from cognee.infrastructure.files.storage import get_file_storage
from cognee.base_config import get_base_config

# Non-critical deps
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)
from cognee.api.v1.permissions.routers import get_permissions_router
from cognee.api.v1.settings.routers import get_settings_router
from cognee.api.v1.datasets.routers import get_datasets_router
from cognee.api.v1.cognify.routers import get_code_pipeline_router, get_cognify_router
from cognee.api.v1.search.routers import get_search_router
from cognee.api.v1.add.routers import get_add_router
from cognee.api.v1.delete.routers import get_delete_router
from cognee.api.v1.responses.routers import get_responses_router
from cognee.api.v1.users.routers import (
    get_auth_router,
    get_register_router,
    get_reset_password_router,
    get_verify_router,
    get_users_router,
    get_visualize_router,
)

logger = get_logger()

if os.getenv("ENV", "prod") == "prod":
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_REPORTING_URL"),
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )


app_environment = os.getenv("ENV", "prod")
app_started_at = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # from cognee.modules.data.deletion import prune_system, prune_data
    # await prune_data()
    # await prune_system(metadata = True)
    # if app_environment == "local" or app_environment == "dev":
    from cognee.infrastructure.databases.relational import get_relational_engine

    db_engine = get_relational_engine()
    await db_engine.create_database()

    from cognee.modules.users.methods import get_default_user

    await get_default_user()

    yield


app = FastAPI(debug=app_environment != "prod", lifespan=lifespan)


# Read allowed origins from environment variable (comma-separated)
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS")
if CORS_ALLOWED_ORIGINS:
    allowed_origins = [
        origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(",") if origin.strip()
    ]
else:
    allowed_origins = [
        "http://localhost:3000",
    ]  # Block all except explicitly set origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Now controlled by env var
    allow_credentials=True,
    allow_methods=["OPTIONS", "GET", "POST", "DELETE"],
    allow_headers=["*"],
)
# To allow origins, set CORS_ALLOWED_ORIGINS env variable to a comma-separated list, e.g.:
# CORS_ALLOWED_ORIGINS="https://yourdomain.com,https://another.com"


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Cognee API",
        version="1.0.0",
        description="Cognee API with Bearer token and Cookie auth",
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer"},
        "CookieAuth": {
            "type": "apiKey",
            "in": "cookie",
            "name": os.getenv("AUTH_TOKEN_COOKIE_NAME", "auth_token"),
        },
    }

    openapi_schema["security"] = [{"BearerAuth": []}, {"CookieAuth": []}]

    app.openapi_schema = openapi_schema

    return app.openapi_schema


app.openapi = custom_openapi


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path == "/api/v1/auth/login":
        return JSONResponse(
            status_code=400,
            content={"detail": "LOGIN_BAD_CREDENTIALS"},
        )

    return JSONResponse(
        status_code=400,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.exception_handler(CogneeApiError)
async def exception_handler(_: Request, exc: CogneeApiError) -> JSONResponse:
    detail = {}

    if exc.name and exc.message and exc.status_code:
        status_code = exc.status_code
        detail["message"] = f"{exc.message} [{exc.name}]"
    else:
        # Log an error indicating the exception is improperly defined
        logger.error("Improperly defined exception: %s", exc)
        # Provide a default error response
        detail["message"] = "An unexpected error occurred."
        status_code = status.HTTP_418_IM_A_TEAPOT

    # log the stack trace for easier serverside debugging
    logger.error(format_exc())
    return JSONResponse(status_code=status_code, content={"detail": detail["message"]})


@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Hello, World, I am alive!"}


@app.get("/health")
def health_check():
    """
    Health check endpoint that returns the server status.
    """
    return Response(status_code=200)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uptime_seconds() -> int:
    return int((datetime.now(timezone.utc) - app_started_at).total_seconds())


async def _check_relational_db() -> dict:
    started = time.perf_counter()
    provider = get_relational_config().db_provider or "unknown"
    try:
        db = get_relational_engine()
        # Simple connectivity check
        async with db.engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "healthy",
            "provider": provider,
            "response_time_ms": duration,
            "details": "Connection successful",
        }
    except Exception as exc:
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "unhealthy",
            "provider": provider,
            "response_time_ms": duration,
            "details": f"Relational DB check failed: {type(exc).__name__}",
        }


async def _check_vector_db() -> dict:
    started = time.perf_counter()
    provider = get_vectordb_config().vector_db_provider or "unknown"
    try:
        engine = get_vector_engine()
        # Try a lightweight operation that touches the backend
        if hasattr(engine, "has_collection"):
            _ = await engine.has_collection("_healthcheck")
        elif hasattr(engine, "get_connection"):
            conn = await engine.get_connection()  # noqa: F841
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "healthy",
            "provider": provider,
            "response_time_ms": duration,
            "details": "Vector DB reachable",
        }
    except Exception as exc:
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "unhealthy",
            "provider": provider,
            "response_time_ms": duration,
            "details": f"Vector DB check failed: {type(exc).__name__}",
        }


async def _check_graph_db() -> dict:
    started = time.perf_counter()
    provider = graph_config_module.get_graph_config().graph_database_provider or "unknown"
    try:
        graph = await get_graph_engine()
        # Use a method defined on the interface to ensure connectivity
        if hasattr(graph, "get_graph_metrics"):
            _ = await graph.get_graph_metrics(include_optional=False)
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "healthy",
            "provider": provider,
            "response_time_ms": duration,
            "details": "Graph DB reachable",
        }
    except Exception as exc:
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "unhealthy",
            "provider": provider,
            "response_time_ms": duration,
            "details": f"Graph DB check failed: {type(exc).__name__}",
        }


async def _check_file_storage() -> dict:
    started = time.perf_counter()
    base_cfg = get_base_config()
    storage_provider = (
        "s3"
        if os.getenv("STORAGE_BACKEND", "").lower() == "s3"
        or base_cfg.system_root_directory.startswith("s3://")
        or base_cfg.data_root_directory.startswith("s3://")
        else "local"
    )
    try:
        storage = get_file_storage(base_cfg.system_root_directory)
        # Attempt to write and remove a tiny temp file to verify permissions
        tmp_name = "healthcheck.tmp"
        await storage.ensure_directory_exists()
        await storage.store(tmp_name, "ok", overwrite=True)
        await storage.remove(tmp_name)
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "healthy",
            "provider": storage_provider,
            "response_time_ms": duration,
            "details": "Storage accessible",
        }
    except Exception as exc:
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "unhealthy",
            "provider": storage_provider,
            "response_time_ms": duration,
            "details": f"File storage check failed: {type(exc).__name__}",
        }


async def _check_llm_provider() -> dict:
    started = time.perf_counter()
    llm_provider = get_llm_config().llm_provider or "unknown"
    try:
        # Instantiate client to validate configuration; avoid making a billable call
        _ = get_llm_client()
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "healthy",
            "provider": llm_provider,
            "response_time_ms": duration,
            "details": "Client configured",
        }
    except Exception as exc:
        duration = int((time.perf_counter() - started) * 1000)
        # Non-critical -> degraded when failing
        return {
            "status": "degraded",
            "provider": llm_provider,
            "response_time_ms": duration,
            "details": f"LLM client initialization failed: {type(exc).__name__}",
        }


async def _check_embedding_service() -> dict:
    started = time.perf_counter()
    emb_provider = get_embedding_config().embedding_provider or "unknown"
    try:
        engine = get_embedding_engine()
        # Attempt a tiny embed; if provider is purely local, this stays in-process
        await engine.embed_text(["ok"])
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "healthy",
            "provider": emb_provider,
            "response_time_ms": duration,
            "details": "Embedding generation working",
        }
    except Exception as exc:
        duration = int((time.perf_counter() - started) * 1000)
        return {
            "status": "unhealthy",
            "provider": emb_provider,
            "response_time_ms": duration,
            "details": f"Embedding check failed: {type(exc).__name__}",
        }


def _aggregate_overall_status(components: dict) -> str:
    critical = [
        components["relational_db"]["status"],
        components["vector_db"]["status"],
        components["graph_db"]["status"],
        components["file_storage"]["status"],
    ]
    if any(status_val == "unhealthy" for status_val in critical):
        return "unhealthy"
    non_critical = [
        components["llm_provider"]["status"],
        components["embedding_service"]["status"],
    ]
    if any(status_val != "healthy" for status_val in non_critical):
        return "degraded"
    return "healthy"


@app.get("/health/ready")
async def readiness_check():
    # Only check critical services for readiness
    relational_db, vector_db, graph_db, file_storage = await asyncio.gather(
        _check_relational_db(), _check_vector_db(), _check_graph_db(), _check_file_storage()
    )

    components = {
        "relational_db": relational_db,
        "vector_db": vector_db,
        "graph_db": graph_db,
        "file_storage": file_storage,
        # Non-critical omitted in readiness
    }

    overall_status = (
        "unhealthy"
        if any(c["status"] == "unhealthy" for c in components.values())
        else "healthy"
    )

    payload = {
        "status": overall_status,
        "timestamp": _now_utc_iso(),
        "version": get_cognee_version(),
        "uptime": _uptime_seconds(),
        "components": components,
    }

    if overall_status == "unhealthy":
        return JSONResponse(status_code=503, content=payload)
    return JSONResponse(status_code=200, content=payload)


@app.get("/health/detailed")
async def detailed_health():
    # Run all checks in parallel
    (
        relational_db,
        vector_db,
        graph_db,
        file_storage,
        llm_provider,
        embedding_service,
    ) = await asyncio.gather(
        _check_relational_db(),
        _check_vector_db(),
        _check_graph_db(),
        _check_file_storage(),
        _check_llm_provider(),
        _check_embedding_service(),
    )

    components = {
        "relational_db": relational_db,
        "vector_db": vector_db,
        "graph_db": graph_db,
        "file_storage": file_storage,
        "llm_provider": llm_provider,
        "embedding_service": embedding_service,
    }

    overall_status = _aggregate_overall_status(components)
    payload = {
        "status": overall_status,
        "timestamp": _now_utc_iso(),
        "version": get_cognee_version(),
        "uptime": _uptime_seconds(),
        "components": components,
    }

    # Non-critical failures should not cause non-200 here per requirements
    if overall_status == "unhealthy":
        return JSONResponse(status_code=503, content=payload)
    return JSONResponse(status_code=200, content=payload)


app.include_router(get_auth_router(), prefix="/api/v1/auth", tags=["auth"])

app.include_router(
    get_register_router(),
    prefix="/api/v1/auth",
    tags=["auth"],
)

app.include_router(
    get_reset_password_router(),
    prefix="/api/v1/auth",
    tags=["auth"],
)

app.include_router(
    get_verify_router(),
    prefix="/api/v1/auth",
    tags=["auth"],
)

app.include_router(get_add_router(), prefix="/api/v1/add", tags=["add"])

app.include_router(get_cognify_router(), prefix="/api/v1/cognify", tags=["cognify"])

app.include_router(get_search_router(), prefix="/api/v1/search", tags=["search"])

app.include_router(
    get_permissions_router(),
    prefix="/api/v1/permissions",
    tags=["permissions"],
)

app.include_router(get_datasets_router(), prefix="/api/v1/datasets", tags=["datasets"])

app.include_router(get_settings_router(), prefix="/api/v1/settings", tags=["settings"])

app.include_router(get_visualize_router(), prefix="/api/v1/visualize", tags=["visualize"])

app.include_router(get_delete_router(), prefix="/api/v1/delete", tags=["delete"])

app.include_router(get_responses_router(), prefix="/api/v1/responses", tags=["responses"])

codegraph_routes = get_code_pipeline_router()
if codegraph_routes:
    app.include_router(codegraph_routes, prefix="/api/v1/code-pipeline", tags=["code-pipeline"])

app.include_router(
    get_users_router(),
    prefix="/api/v1/users",
    tags=["users"],
)


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server using uvicorn.
    Parameters:
    host (str): The host for the server.
    port (int): The port for the server.
    """
    try:
        logger.info("Starting server at %s:%s", host, port)

        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        # Here you could add any cleanup code or error recovery code.
        raise e


if __name__ == "__main__":
    logger = setup_logging()

    start_api_server(
        host=os.getenv("HTTP_API_HOST", "0.0.0.0"), port=int(os.getenv("HTTP_API_PORT", 8000))
    )
