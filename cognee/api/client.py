"""FastAPI server for the Cognee API."""

import os
import uvicorn
from cognee.shared.logging_utils import get_logger
import sentry_sdk
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from cognee.api.v1.permissions.routers import get_permissions_router
from cognee.api.v1.settings.routers import get_settings_router
from cognee.api.v1.datasets.routers import get_datasets_router
from cognee.api.v1.cognify.routers import get_code_pipeline_router, get_cognify_router
from cognee.api.v1.search.routers import get_search_router
from cognee.api.v1.add.routers import get_add_router
from cognee.api.v1.delete.routers import get_delete_router
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from cognee.exceptions import CogneeApiError
from traceback import format_exc
from cognee.api.v1.users.routers import (
    get_auth_router,
    get_register_router,
    get_reset_password_router,
    get_verify_router,
    get_users_router,
    get_visualize_router,
)
from contextlib import asynccontextmanager

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["OPTIONS", "GET", "POST", "DELETE"],
    allow_headers=["*"],
)


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

app.include_router(
    get_users_router(),
    prefix="/api/v1/users",
    tags=["users"],
)

app.include_router(
    get_permissions_router(),
    prefix="/api/v1/permissions",
    tags=["permissions"],
)


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


app.include_router(get_datasets_router(), prefix="/api/v1/datasets", tags=["datasets"])

app.include_router(get_add_router(), prefix="/api/v1/add", tags=["add"])

app.include_router(get_cognify_router(), prefix="/api/v1/cognify", tags=["cognify"])

app.include_router(get_search_router(), prefix="/api/v1/search", tags=["search"])

app.include_router(get_settings_router(), prefix="/api/v1/settings", tags=["settings"])

app.include_router(get_visualize_router(), prefix="/api/v1/visualize", tags=["visualize"])

app.include_router(get_delete_router(), prefix="/api/v1/delete", tags=["delete"])

codegraph_routes = get_code_pipeline_router()
if codegraph_routes:
    app.include_router(codegraph_routes, prefix="/api/v1/code-pipeline", tags=["code-pipeline"])


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
    start_api_server()
