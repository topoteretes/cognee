from fastapi_users.authentication import BearerTransport

api_bearer_transport = BearerTransport(
    tokenUrl="/api/v1/auth/token",
)

api_bearer_transport.name = "bearer"
