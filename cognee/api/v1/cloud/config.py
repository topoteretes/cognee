from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class CloudConfig(BaseSettings):
    cognee_cloud_api_url: str = "http://localhost:8001"
    cognee_cloud_auth_token: str = "your-auth-token"
    cognee_cloud_url: str = "https://platform.cognee.ai"
    cognee_service_url: Optional[str] = None
    cognee_api_key: Optional[str] = None
    cognee_api_token: Optional[str] = None
    ui_app_url: str = "http://localhost:3000"
    http_api_host: str = "0.0.0.0"
    http_api_port: int = 8000
    cors_allowed_origins: Optional[str] = None
    sentry_reporting_url: Optional[str] = None
    cognee_auth0_domain: Optional[str] = None
    cognee_auth0_device_client_id: str = ""
    cognee_auth0_audience: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "cognee_cloud_api_url": self.cognee_cloud_api_url,
            "cognee_cloud_auth_token": self.cognee_cloud_auth_token,
            "cognee_cloud_url": self.cognee_cloud_url,
            "cognee_service_url": self.cognee_service_url,
            "cognee_api_key": self.cognee_api_key,
            "cognee_api_token": self.cognee_api_token,
            "ui_app_url": self.ui_app_url,
            "http_api_host": self.http_api_host,
            "http_api_port": self.http_api_port,
            "cors_allowed_origins": self.cors_allowed_origins,
            "sentry_reporting_url": self.sentry_reporting_url,
            "cognee_auth0_domain": self.cognee_auth0_domain,
            "cognee_auth0_device_client_id": self.cognee_auth0_device_client_id,
            "cognee_auth0_audience": self.cognee_auth0_audience,
        }


@lru_cache
def get_cloud_config():
    return CloudConfig()
