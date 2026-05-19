from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class TelemetryConfig(BaseSettings):
    telemetry_disabled: bool = False
    telemetry_request_timeout: int = 5
    telemetry_api_key_tracking_salt: Optional[str] = None
    tracking_id: Optional[str] = None
    env: str = "prod"
    cognee_tracing_enabled: bool = False
    otel_service_name: str = "cognee"
    otel_exporter_otlp_endpoint: Optional[str] = None
    otel_exporter_otlp_headers: Optional[str] = None
    otel_resource_attributes: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "telemetry_disabled": self.telemetry_disabled,
            "telemetry_request_timeout": self.telemetry_request_timeout,
            "telemetry_api_key_tracking_salt": self.telemetry_api_key_tracking_salt,
            "tracking_id": self.tracking_id,
            "env": self.env,
            "cognee_tracing_enabled": self.cognee_tracing_enabled,
            "otel_service_name": self.otel_service_name,
            "otel_exporter_otlp_endpoint": self.otel_exporter_otlp_endpoint,
            "otel_exporter_otlp_headers": self.otel_exporter_otlp_headers,
            "otel_resource_attributes": self.otel_resource_attributes,
        }


@lru_cache
def get_telemetry_config():
    return TelemetryConfig()
