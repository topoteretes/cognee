"""NuGet v3 package registry connector for Cognee."""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from .source import nuget_source

CONNECTOR = ConnectorDescriptor(
    provider="nuget",
    display_name="NuGet",
    auth="none",
)

__all__ = ["CONNECTOR", "nuget_source"]
