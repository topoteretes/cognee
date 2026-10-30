"""The connector registry, and the entry-point loaders hosts call.

Moved from cloud-backend ``modules.integrations.framework.registry``
(cognee-saas-backend, 2026-10-26), minus the cloud's reserved provider name
and secret-blob availability check; the entry-point loaders are new and are
how a host discovers installed connector packages without importing them by
name.

A plain dict, keyed by provider. Mirrors the shape cognee uses for its own
pluggable seams (``use_vector_adapter``/``supported_databases``, and
``use_integration``) so the pattern is already familiar.

**Registration is not availability.** A connector is registered when its
package is installed; it is *available* when the host also has its settings.
Those differ per deployment, so the catalog reports what is registered and a
host decides what it can serve: an unregistered provider is 404 (it does not
exist), an unconfigured one is 503 (it exists, not here, not yet).

**Why the registry does not import providers itself.** Connector packages are
found through entry points (see :func:`load_connectors`), each loaded on its
own so a broken or half-finished package cannot take the registry — and
therefore every other connector — down with it. A package that is never
installed simply never registers.
"""

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Any

from cognee_connectors_core.base import Connector
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

logger = logging.getLogger(__name__)

#: Entry-point group naming each package's contract object (``CONNECTOR``).
CONNECTORS_GROUP = "cognee.connectors"
#: Entry-point group naming each package's DLT source factories.
SOURCES_GROUP = "cognee.sources"
#: Entry-point group naming each package's cognee-host ``register()``.
INTEGRATIONS_GROUP = "cognee.integrations"

# Matches the length cap a sync payload's provider field enforces elsewhere
# (cognee-saas-pod's ConnectorSyncPayload.provider, Field(max_length=100)) —
# chosen there, not here, so this is a cross-reference, not an independent
# decision.
_MAX_PROVIDER_LENGTH = 100

_connectors: dict[str, Connector] = {}

# Tracks the settings-normalized form (upper-cased, hyphens to underscores) of
# every registered provider, so a second provider whose raw name normalizes
# the same way is caught here rather than discovered as "two providers are
# reading the same {PROVIDER}_* settings". Separate from _connectors's own
# keys, which stay raw (that dict is the actual routing table and providers
# must remain distinguishable there).
_normalized_names: dict[str, str] = {}


def _normalized(provider: str) -> str:
    """The form a provider's settings ``env_prefix`` is derived from.

    Two providers differing only in upper/lower-case or ``-``/``_`` are caught
    as the same settings consumer, which is the actual collision
    (``ms-teams``/``ms_teams`` both resolving to ``MS_TEAMS_``), not just
    visually similar names.
    """
    return provider.upper().replace("-", "_")


def use_connector(connector: Connector | ConnectorDescriptor) -> Connector:
    """Register a connector, replacing any previous one for its provider.

    A bare :class:`~.descriptor.ConnectorDescriptor` is accepted and wrapped in
    a :class:`~.oauth2.DescriptorConnector` without settings — enough for an
    API-key or local provider, whose descriptor only names things.

    Replacement is intentional and is what lets a Tier 3 implementation
    supersede a descriptor-driven one for the same provider — GitHub can ship
    as a descriptor first and be upgraded in place. It is logged, because a
    replacement nobody intended is otherwise invisible.
    """
    if isinstance(connector, ConnectorDescriptor):
        connector = DescriptorConnector(connector)
    if not isinstance(connector, Connector):
        raise TypeError(
            f"use_connector() needs a Connector or ConnectorDescriptor, got {connector!r}"
        )

    provider = getattr(connector, "provider", None)
    if not provider:
        raise ValueError(f"{connector!r} has no provider set")
    if len(provider) > _MAX_PROVIDER_LENGTH:
        # An arbitrarily long provider name is a registry bug, not a real
        # connector, so it is refused here, once, before anything is ever
        # written under it.
        raise ValueError(
            f"provider name {provider!r} is {len(provider)} chars, over the "
            f"{_MAX_PROVIDER_LENGTH}-char limit a connector-sync payload enforces"
        )
    if ":" in provider:
        # oauth_state's signed state is colon-joined and validate_state splits
        # on field count. A colon in the provider name changes the field count
        # of every state that carries it, so it fails CLOSED — the state never
        # validates and the connector can never complete an install — rather
        # than corrupting another field. Still refused at registration: "this
        # connector's OAuth flow silently never works" is a bad way to
        # discover a naming mistake.
        raise ValueError(
            f"provider name {provider!r} cannot contain ':' — it is a field separator "
            "in the signed OAuth state encoding"
        )
    if not provider.isascii() or not _normalized(provider).replace("_", "").isalnum():
        # A provider's settings prefix is {NORMALIZED}_ and an env var name
        # cannot contain a space, slash or dot at all, so a name with one
        # survives registration and can never actually be configured. Same
        # silent-permanently-unconfigured shape the ':' check above exists to
        # catch, just for a different separator.
        #
        # isascii() is what makes isalnum() mean what this check needs it to.
        # str.isalnum() is Unicode-aware, so "café", "провайдер" and the
        # fullwidth "ｇithub" all pass it and all normalize to env var names
        # that are not valid either.
        raise ValueError(
            f"provider name {provider!r} must be ASCII alphanumeric (with '-'/'_' "
            f"allowed) — anything else cannot become the {_normalized(provider)!r}-shaped "
            "settings prefix"
        )

    normalized = _normalized(provider)
    colliding = _normalized_names.get(normalized)
    if colliding is not None and colliding != provider:
        # "ms-teams" and "ms_teams" both resolve to MS_TEAMS_. Two
        # registry-distinct providers would then read the same settings,
        # meaning one's webhook could verify under the other's key. Refused
        # here rather than left to be found the day both actually ship.
        raise ValueError(
            f"provider name {provider!r} collides with already-registered "
            f"{colliding!r} once normalized for settings lookup (both resolve to "
            f"{normalized}_)"
        )
    _normalized_names[normalized] = provider

    if provider in _connectors:
        logger.info(
            "Replacing connector for %s: %r -> %r", provider, _connectors[provider], connector
        )

    try:
        has_verifier = connector.webhook_verifier() is not None
    except Exception:
        # A connector that cannot even describe itself is reported as degraded
        # by catalog() — registration must not fail a second, harsher way over
        # the same problem, but the traceback is worth having once.
        logger.warning(
            "%s failed to build its webhook verifier at registration", provider, exc_info=True
        )
        has_verifier = False

    if has_verifier and type(connector).handle_webhook is Connector.handle_webhook:
        # webhook_verifier() and handle_webhook() are independent optional
        # hooks — a connector overriding only the former passes signature
        # verification and then hits the base class's NotImplementedError.
        # Same is-this-overridden check supports_resource_selection uses.
        # Warn rather than raise: a connector mid-migration to webhook support
        # can legitimately have one without the other for a short stretch.
        logger.warning(
            "%s declares a webhook_verifier() but has not overridden handle_webhook() — "
            "every verified delivery will hit NotImplementedError",
            provider,
        )

    _connectors[provider] = connector
    return connector


def get_connector(provider: str) -> Connector:
    """The connector for ``provider``.

    Raises ``KeyError``, which a host turns into 404. Deliberately not a
    custom exception: a host maps a missing provider and a provider with no
    webhook verifier to the same 404, so that a caller cannot use the
    difference to enumerate which providers exist.
    """
    return _connectors[provider]


def find_connector(provider: str) -> Connector | None:
    """The connector for ``provider``, or ``None``. For non-request callers."""
    return _connectors.get(provider)


def registered_providers() -> list[str]:
    """Every registered provider name, sorted for stable output."""
    return sorted(_connectors)


def tier_of(connector: Connector) -> int:
    """A guess at the connector's tier, from how it is built.

    1 for a bare :class:`~.oauth2.DescriptorConnector`, 2 for a subclass of it
    (a descriptor plus a hook), 3 for anything else on the contract. A guess
    because the tier is an observation about a provider, not a declaration —
    but the class a connector is built from is the only evidence a catalog
    has, and it is right for every provider on the roadmap.
    """
    if not isinstance(connector, DescriptorConnector):
        return 3
    if type(connector) is DescriptorConnector:
        return 1
    return 2


def describe(connector: Connector) -> dict[str, Any]:
    """One catalog entry. Separate so a failure is attributable to one connector."""
    return {
        "provider": connector.provider,
        "display_name": connector.display_name,
        "auth": connector.auth,
        "tier": tier_of(connector),
        "has_webhooks": connector.webhook_verifier() is not None,
        # So the UI knows whether to offer a picker without probing
        # GET /{provider}/resources and reading a 404 as the answer.
        "supports_resource_selection": type(connector).supports_resource_selection(),
        "poll_interval_seconds": connector.poll_interval_seconds,
        "content_type": connector.content_type,
        "degraded": False,
    }


def catalog() -> list[dict[str, Any]]:
    """The connector catalog, for a status route and the UI.

    Display facts only. Nothing here is derived from a credential or a secret
    value, so this is safe to serve to any authenticated caller; whether a
    provider is *configured* on this deployment is the host's to add.

    Each connector is described independently: one that raises while describing
    itself is reported as degraded rather than taking the response with it. That
    matters at 50 connectors, where a single typo in provider #37's descriptor
    would otherwise blank the status page for everyone looking at any of the
    other 49.
    """
    entries = []
    for provider in registered_providers():
        connector = _connectors[provider]
        try:
            entries.append(describe(connector))
        except Exception:
            logger.exception("Connector %s failed to describe itself", provider)
            entries.append(
                {
                    "provider": provider,
                    "display_name": provider,
                    "auth": getattr(connector, "auth", "oauth2"),
                    "tier": tier_of(connector),
                    "has_webhooks": False,
                    "supports_resource_selection": False,
                    "poll_interval_seconds": None,
                    "content_type": None,
                    # Surfaced so a misconfigured connector is visibly broken
                    # rather than quietly reporting itself as unconfigured —
                    # those need different fixes.
                    "degraded": True,
                }
            )
    return entries


def reset_registry() -> None:
    """Empty the registry. For tests that register throwaway connectors."""
    _connectors.clear()
    _normalized_names.clear()


# ---- Entry points -------------------------------------------------------


def _entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    """Every installed entry point in ``group``.

    ``entry_points(group=...)`` is the selectable API Python 3.10 added; the
    fallback reads the older shape, a mapping of group name to entry points,
    which is also what some backports return.
    """
    try:
        selected = importlib.metadata.entry_points(group=group)
    except TypeError:
        selected = importlib.metadata.entry_points().get(group, [])  # type: ignore[union-attr]
    return list(selected)


def load_connectors(group: str = CONNECTORS_GROUP) -> dict[str, Connector]:
    """Register every connector an installed package advertises, by provider.

    Each entry point is loaded and registered on its own; one that fails to
    import or to register is logged and skipped, so a broken package cannot
    hide the others. The entry point's name should be the provider name — a
    mismatch is logged, and the provider name wins.
    """
    loaded: dict[str, Connector] = {}
    for entry_point in _entry_points(group):
        try:
            connector = use_connector(entry_point.load())
        except Exception:
            logger.exception(
                "Skipping %s entry point %r (%s): it failed to load or register",
                group,
                entry_point.name,
                entry_point.value,
            )
            continue
        if entry_point.name != connector.provider:
            logger.warning(
                "%s entry point %r registers provider %r; the names should match",
                group,
                entry_point.name,
                connector.provider,
            )
        loaded[connector.provider] = connector
    return loaded


def load_sources(group: str = SOURCES_GROUP) -> dict[str, Callable[..., Any]]:
    """Every DLT source factory an installed package advertises, by entry-point name.

    Nothing is called: a factory takes provider-specific arguments (a token,
    a selection) that only the caller knows. An entry point that fails to
    load, or names something that is not callable, is logged and skipped.
    """
    loaded: dict[str, Callable[..., Any]] = {}
    for entry_point in _entry_points(group):
        try:
            source = entry_point.load()
        except Exception:
            logger.exception(
                "Skipping %s entry point %r (%s): it failed to load",
                group,
                entry_point.name,
                entry_point.value,
            )
            continue
        if not callable(source):
            logger.warning(
                "Skipping %s entry point %r (%s): %r is not callable",
                group,
                entry_point.name,
                entry_point.value,
                source,
            )
            continue
        loaded[entry_point.name] = source
    return loaded


def load_host_registrations(group: str = INTEGRATIONS_GROUP) -> list[str]:
    """Call every installed package's cognee-host ``register()``; return who registered.

    The entry point names a zero-argument callable
    (``cognee_connector_<p>.cognee_host:register``) that plugs the provider
    into cognee's own integration registry. It needs the package's ``cognee``
    extra: a package installed without it fails to import here, which is
    logged at warning level — the expected shape for an SDK-only install, not
    an error — and skipped. The returned names are the entry points whose
    ``register()`` ran without raising.
    """
    registered: list[str] = []
    for entry_point in _entry_points(group):
        try:
            register = entry_point.load()
            register()
        except Exception as error:
            logger.warning(
                "Skipping %s registration %r (%s): %s",
                group,
                entry_point.name,
                entry_point.value,
                error,
            )
            logger.debug("Registration %r failed", entry_point.name, exc_info=True)
            continue
        registered.append(entry_point.name)
    return registered


__all__ = [
    "CONNECTORS_GROUP",
    "INTEGRATIONS_GROUP",
    "SOURCES_GROUP",
    "catalog",
    "describe",
    "find_connector",
    "get_connector",
    "load_connectors",
    "load_host_registrations",
    "load_sources",
    "registered_providers",
    "reset_registry",
    "tier_of",
    "use_connector",
]
