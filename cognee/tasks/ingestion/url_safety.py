"""Safety guards for ingesting remote URLs.

``save_data_item_to_storage`` will fetch any ``http(s)`` string handed to it.
Without guards this is a Server-Side Request Forgery (SSRF) primitive: a caller
can make the server request internal-only endpoints (cloud metadata at
``169.254.169.254``, ``localhost`` admin panels, RFC-1918 hosts) and have the
response ingested and later retrievable via search. The documented
``ALLOW_HTTP_REQUESTS`` gate was also never actually enforced.
"""

import ipaddress
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from cognee.modules.ingestion.exceptions import IngestionError


def _http_requests_allowed() -> bool:
    return os.getenv("ALLOW_HTTP_REQUESTS", "true").lower() == "true"


def assert_url_allowed(url: str) -> None:
    """Raise ``IngestionError`` if ``url`` must not be fetched.

    Enforces ``ALLOW_HTTP_REQUESTS`` and blocks any host that resolves to a
    private, loopback, link-local, reserved, multicast or unspecified address.
    """
    if not _http_requests_allowed():
        raise IngestionError(message="HTTP requests are disabled (ALLOW_HTTP_REQUESTS=false).")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IngestionError(message=f"Unsupported URL scheme for fetching: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise IngestionError(message="URL has no host.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addr_infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as error:
        raise IngestionError(message=f"Could not resolve host: {host}") from error

    for info in addr_infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise IngestionError(
                message=f"Refusing to fetch non-public address for host {host!r}: {ip}"
            )


def _allowed_local_roots() -> list[str]:
    """Directories that local files may be ingested from.

    Overridable via the comma-separated ``LOCAL_FILE_ALLOWED_ROOTS`` env var. The
    default set covers legitimate use (cognee's data directory, the working
    directory, and the system temp dir) while excluding sensitive locations like
    ``/etc``, ``/root``, ``~/.ssh`` and ``/proc``.
    """
    env = os.getenv("LOCAL_FILE_ALLOWED_ROOTS")
    if env:
        raw_roots = [r.strip() for r in env.split(",") if r.strip()]
    else:
        # Imported lazily to avoid a heavy import at module load.
        from cognee.base_config import get_base_config

        raw_roots = [
            get_base_config().data_root_directory,
            os.getcwd(),
            tempfile.gettempdir(),
        ]

    resolved: list[str] = []
    for root in raw_roots:
        # Skip non-local roots (e.g. s3:// data directories); realpath them so
        # symlinked roots compare correctly.
        if "://" in root:
            continue
        resolved.append(os.path.realpath(root))
    return resolved


def assert_local_path_allowed(path: str) -> None:
    """Raise ``IngestionError`` if ``path`` resolves outside the allowed roots.

    ``os.path.realpath`` canonicalises the path, so symlinks that point outside an
    allowed root are rejected rather than followed.
    """
    resolved = os.path.realpath(path)
    for root in _allowed_local_roots():
        try:
            Path(resolved).relative_to(root)
            return
        except ValueError:
            continue

    raise IngestionError(
        message=(
            f"Refusing to ingest local file outside the allowed roots: {path!r}. "
            "Set LOCAL_FILE_ALLOWED_ROOTS to permit additional directories."
        )
    )
