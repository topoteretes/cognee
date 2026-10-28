"""cognee-connectors-core: the contract every cognee connector implements.

No host in here. A connector package depends on this and on its vendor SDK;
cognee and cloud-backend depend on this to load and drive the connectors. See
each module's docstring for the piece it owns:

- :mod:`.base` — ``Connector`` and the dataclasses it speaks in.
- :mod:`.descriptor` — ``ConnectorDescriptor``, a Tier 1 connector as data.
- :mod:`.oauth2` — ``DescriptorConnector``, which makes a descriptor work.
- :mod:`.signatures` — webhook HMAC schemes and ``HmacVerifier``.
- :mod:`.oauth_state` — OAuth ``state`` signing and validation.
- :mod:`.settings` — ``ConnectorSettings`` and ``require``.
- :mod:`.dlt_conventions` — what cognee's ingestion reads off a DLT source.
- :mod:`.registry` — ``use_connector``, ``catalog`` and the entry-point loaders.
- :mod:`.http` — the ``HttpClient`` protocol and its aiohttp default.
- :mod:`.testing` — ``FakeHttpClient`` and fixtures for connector tests.
"""

from cognee_connectors_core import dlt_conventions
from cognee_connectors_core.base import (
    AUTH_KINDS,
    AuthKind,
    Connector,
    ConnectorInstallation,
    ConnectorResource,
    Credential,
    RefreshedTokens,
    SyncScope,
    WebhookDispatch,
    WebhookVerifier,
    account_slug,
    dataset_name_for,
)
from cognee_connectors_core.descriptor import ConnectorDescriptor, payload_paths, resolve_path
from cognee_connectors_core.errors import ConnectorError, RateLimitedError, WebhookVerificationError
from cognee_connectors_core.http import AiohttpClient, HttpClient
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.oauth_state import (
    DEFAULT_STATE_TTL_SECONDS,
    make_state,
    sign_state_payload,
    validate_state,
)
from cognee_connectors_core.registry import (
    catalog,
    find_connector,
    get_connector,
    load_connectors,
    load_host_registrations,
    load_sources,
    registered_providers,
    reset_registry,
    use_connector,
)
from cognee_connectors_core.settings import ConnectorSettings, require
from cognee_connectors_core.signatures import HmacVerifier, SignatureScheme

__version__ = "0.1.0"

__all__ = [
    "AUTH_KINDS",
    "DEFAULT_STATE_TTL_SECONDS",
    "AiohttpClient",
    "AuthKind",
    "Connector",
    "ConnectorDescriptor",
    "ConnectorError",
    "ConnectorInstallation",
    "ConnectorResource",
    "ConnectorSettings",
    "Credential",
    "DescriptorConnector",
    "HmacVerifier",
    "HttpClient",
    "RateLimitedError",
    "RefreshedTokens",
    "SignatureScheme",
    "SyncScope",
    "WebhookDispatch",
    "WebhookVerificationError",
    "WebhookVerifier",
    "__version__",
    "account_slug",
    "catalog",
    "dataset_name_for",
    "dlt_conventions",
    "find_connector",
    "get_connector",
    "load_connectors",
    "load_host_registrations",
    "load_sources",
    "make_state",
    "payload_paths",
    "registered_providers",
    "require",
    "reset_registry",
    "resolve_path",
    "sign_state_payload",
    "use_connector",
    "validate_state",
]
