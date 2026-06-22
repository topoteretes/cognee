from typing import Optional, Dict
from fastapi import Response
from fastapi.security import APIKeyHeader
from fastapi_users.authentication import Transport

from fastapi import Request


class APIKeyHeaderTransport(Transport):
    def __init__(self, header_name: str = "X-Api-Key"):
        self.header_name = header_name

    @property
    def scheme(self) -> APIKeyHeader:
        # Used in OpenAPI; not security type
        return APIKeyHeader(name=self.header_name, auto_error=False)

    async def get_authentication_token(self, request: Request) -> Optional[str]:
        return request.headers.get(self.header_name)

    async def get_login_response(self, token: str, response: Response) -> Response:
        # No login response for API key auth — just return unchanged
        return response

    async def get_logout_response(self, response: Response) -> Response:
        # No logout response for API key auth — just return unchanged
        return response

    def get_openapi_login_responses_success(self) -> Dict[str, Dict]:
        # Not applicable — API key doesn't use login endpoint
        return {}

    def get_openapi_logout_responses_success(self) -> Dict[str, Dict]:
        # Not applicable — API key doesn't use logout endpoint
        return {}


def get_api_key_transport() -> Transport:
    api_key_transport = APIKeyHeaderTransport()

    api_key_transport.name = "apikey"  # type: ignore

    return api_key_transport
