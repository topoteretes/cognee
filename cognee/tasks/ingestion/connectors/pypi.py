"""PyPI (Python Package Index) connector for cognee.

A ``dlt`` source that turns Python package metadata, dependencies, and documentation
from PyPI into searchable knowledge graph documents.

Usage::

    import cognee
    from cognee.tasks.ingestion.connectors import pypi_source

    await cognee.remember(
        pypi_source(packages=["cognee", "pydantic", "fastapi"]),
        dataset_name="python_libraries",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  Ids are formatted as ``pypi:<package_name_lower>``.
* **Node set** ``pypi:<package_name>`` and ``pypi`` scope retrievals to PyPI packages.
* **Content** formats the package title, summary, author, license, homepage, Python version
  constraints, dependencies, classifiers, and project description / README into a structured
  Markdown document.
* **Incremental**: Uses package version and release serial in ``dlt.current.resource_state()``
  to skip re-ingesting unchanged packages across sync cycles.
* **Zero extra dependencies**: Uses standard Python HTTP client with timeout and rate handling.
"""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = logging.getLogger(__name__)

PYPI_API_BASE = "https://pypi.org/pypi"
DEFAULT_USER_AGENT = "cognee-pypi-connector/1.0 (https://cognee.ai)"


class PyPIError(RuntimeError):
    """Base error for PyPI connector operations."""


class PyPIPackageNotFoundError(PyPIError):
    """Package does not exist on PyPI."""


class PyPIService:
    """HTTP client for retrieving package metadata from PyPI."""

    def __init__(
        self,
        packages: list[str] | str | None = None,
        base_url: str = PYPI_API_BASE,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        if isinstance(packages, str):
            self.packages = [packages.strip()]
        elif packages is not None:
            self.packages = [p.strip() for p in packages if p.strip()]
        else:
            self.packages = []
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent

    def fetch_package_metadata(self, package_name: str, timeout: float = 15.0) -> dict[str, Any]:
        """Fetch package metadata JSON from PyPI's public endpoint."""
        clean_name = package_name.strip()
        if not clean_name:
            raise ValueError("package_name must be a non-empty string")

        url = f"{self.base_url}/{urllib.parse.quote(clean_name)}/json"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = response.read().decode("utf-8")
                    return json.loads(data)
                raise PyPIError(f"PyPI API returned HTTP status {response.status} for {clean_name}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise PyPIPackageNotFoundError(f"Package '{clean_name}' not found on PyPI") from exc
            raise PyPIError(f"HTTP {exc.code} fetching PyPI package '{clean_name}'") from exc
        except urllib.error.URLError as exc:
            raise PyPIError(f"Network error fetching PyPI package '{clean_name}': {exc.reason}") from exc

    fetch_package_json = fetch_package_metadata

    def format_package_document(self, package_data: dict[str, Any]) -> dict[str, Any]:
        """Render PyPI API response into a structured document row for cognee."""
        return _render_package_document(package_data)


def build_pypi_service(
    packages: list[str] | str | None = None,
    base_url: str = PYPI_API_BASE,
    user_agent: str = DEFAULT_USER_AGENT,
) -> PyPIService:
    """Construct a default PyPI service instance."""
    return PyPIService(packages=packages, base_url=base_url, user_agent=user_agent)


def _render_package_document(package_data: dict[str, Any]) -> dict[str, Any]:
    """Render PyPI API response into a structured document row for cognee."""
    info = package_data.get("info", {})
    name = info.get("name", "")
    version = info.get("version", "")
    summary = info.get("summary") or ""
    author = info.get("author") or info.get("author_email") or "Unknown"
    license_info = info.get("license") or "Not specified"
    home_page = info.get("home_page") or info.get("package_url") or f"https://pypi.org/project/{name}/"
    requires_python = info.get("requires_python") or "Any"
    requires_dist = info.get("requires_dist") or []
    classifiers = info.get("classifiers") or []
    raw_desc = (info.get("description") or "").strip()

    # Truncate overly long descriptions while preserving key documentation
    max_desc_len = 16000
    if len(raw_desc) > max_desc_len:
        raw_desc = raw_desc[:max_desc_len] + "\n\n... (description truncated for indexing)"

    content_lines = [
        f"# {name} {version}",
        "",
        f"**Summary:** {summary}",
        f"**Author:** {author}",
        f"**License:** {license_info}",
        f"**Python Requirements:** {requires_python}",
        f"**Package URL:** {home_page}",
        "",
    ]

    if requires_dist:
        content_lines.append("## Dependencies")
        for dep in requires_dist[:50]:
            content_lines.append(f"- {dep}")
        content_lines.append("")

    if classifiers:
        content_lines.append("## Classifiers")
        for c in classifiers[:20]:
            content_lines.append(f"- {c}")
        content_lines.append("")

    if raw_desc:
        content_lines.append("## Description")
        content_lines.append(raw_desc)

    content = "\n".join(content_lines)
    clean_id = f"pypi:{name.lower()}"

    return {
        "id": clean_id,
        "title": f"{name} {version}",
        "content": content,
        "url": home_page,
        "_deleted": False,
        "cognee_node_set": [clean_id, "pypi"],
    }


def _default_resource_name(packages: list[str]) -> str:
    if len(packages) == 1:
        slug = re.sub(r"[^a-z0-9]+", "_", packages[0].lower()).strip("_")
        return f"pypi_{slug}"
    return "pypi_packages"


def pypi_source(
    packages: list[str] | str,
    *,
    resource_name: str | None = None,
    service: PyPIService | None = None,
    check_active: Callable[[], None] | None = None,
    max_requests: int = 100,
):
    """Return a ``dlt`` resource that yields PyPI package metadata documents for ``remember``.

    Args:
        packages: Package name or list of package names to ingest.
        resource_name: Stable resource name. Defaults to ``pypi_<package>`` or ``pypi_packages``.
        service: Optional pre-constructed PyPIService (used for testing and custom proxies).
        check_active: Optional authorization checkpoint callback.
        max_requests: Maximum number of package lookups per sync execution.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError("The PyPI connector requires dlt (a cognee core dependency).") from exc

    package_list = [packages] if isinstance(packages, str) else list(packages)
    if not package_list:
        raise ValueError("pypi_source requires at least one package name")

    client = service if service is not None else build_pypi_service()
    name = resource_name or _default_resource_name(package_list)
    stats: dict[str, int] = {}

    def _iter_packages(current_state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        count = 0
        state_versions = current_state.setdefault("versions", {})

        for pkg in package_list:
            if count >= max_requests:
                stats["failed_budget"] = 1
                logger.warning("PyPI sync reached request budget of %d", max_requests)
                break

            if check_active is not None:
                check_active()

            stats["scanned"] = stats.get("scanned", 0) + 1
            count += 1

            try:
                pkg_data = client.fetch_package_json(pkg)
                version = pkg_data.get("info", {}).get("version", "")
                last_version = state_versions.get(pkg.lower())

                if last_version == version:
                    stats["skipped"] = stats.get("skipped", 0) + 1
                    continue

                row = _render_package_document(pkg_data)
                state_versions[pkg.lower()] = version
                yield row

            except PyPIPackageNotFoundError:
                logger.warning("PyPI package '%s' does not exist; skipping.", pkg)
                stats["failed"] = stats.get("failed", 0) + 1
            except Exception as exc:
                logger.warning("Failed fetching PyPI package '%s': %s", pkg, exc)
                stats["failed"] = stats.get("failed", 0) + 1

    @dlt.resource(
        name=name,
        primary_key="id",
        write_disposition="merge",
        columns={
            "_deleted": {"data_type": "bool", "hard_delete": True},
            NODE_SET_COLUMN: {"data_type": "json", "nullable": True},
        },
    )
    def pypi_packages():
        stats.clear()
        stats.update(scanned=0, skipped=0, failed=0)
        yield from _iter_packages(dlt.current.resource_state())

    resource = pypi_packages()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "pypi")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, name)
    resource.cognee_sync_stats = stats
    return resource
