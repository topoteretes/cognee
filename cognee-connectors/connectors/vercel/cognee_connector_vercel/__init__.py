"""Vercel connector and DLT source; importing this package needs no Cognee host."""

from cognee_connector_vercel.connector import CONNECTOR
from cognee_connector_vercel.source import build_vercel_service, vercel_source

__all__ = ["CONNECTOR", "build_vercel_service", "vercel_source"]
