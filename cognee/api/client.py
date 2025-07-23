"""FastAPI server for the Cognee API."""

import os

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

from cognee.exceptions import CogneeApiError
from cognee.shared.logging_utils import get_logger, setup_logging
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
    allowed_origins = []  # Block all except explicitly set origins

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
