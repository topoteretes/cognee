"""Hex.pm connector for cognee: a ``dlt`` source that turns Hex packages into memory.

It yields one document per package and one per release (with its requirements and any
retirement notice) as flat rows, and is meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import hexpm_source

    await cognee.remember(
        hexpm_source(resource_name="hexpm_deps", package_names=["phoenix", "ecto"]),
        dataset_name="hexpm",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are ``{id, title, content, url, _deleted, cognee_node_set}`` with ids
  ``package:<name>`` and ``release:<name>:<version>``. Rendering is deterministic (sorted
  collections, no volatile counters except the all-time download count of a package row).
* **Node set** ``hexpm:<package name>`` on every row of a package.
* **Selection** is ``package_names`` (e.g. from :func:`parse_mix_lock`), or ``search`` (the
  ``GET /api/packages?search=`` result, ordered by ``updated_at``), or both.
* **Auth** is none for public packages; ``api_key`` and ``repository`` read a private
  organization's ``/api/repos/<repository>/packages/<name>``.
* **Incremental.** Per package the state keeps the ``ETag``, ``updated_at``, the retirements and
  the release versions. A ``304`` or an unchanged ``updated_at`` and retirements fetches nothing
  more; otherwise only new releases and releases whose retirement changed are fetched.
* **Forget-on-delete.** A package that answers 404 (and, for an explicit ``package_names``
  selection, one that left the selection) is tombstoned with all its releases, and a release
  that left the package's release list is tombstoned alone.
"""

import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

BASE_URL = "https://hex.pm/api"
_TIMEOUT = 30
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_LOCK_ENTRY = re.compile(r'^\s*"([a-z][a-z0-9_]*)":\s*\{:hex,', re.MULTILINE)


class HexpmAPIError(RuntimeError):
    """A Hex.pm request that did not answer 200 (or 304 / 404 where those are expected)."""


def parse_mix_lock(text: str) -> list[str]:
    """Names of the Hex packages locked in the text of a ``mix.lock``, sorted."""
    return sorted(set(_LOCK_ENTRY.findall(text)))


class HexpmClient:
    """Minimal Hex.pm JSON client. ``api_key`` is sent as ``Authorization``."""

    def __init__(self, api_key: str | None = None, repository: str | None = None):
        self._api_key = api_key
        self._repository = repository

    def __repr__(self) -> str:
        return f"HexpmClient(repository={self._repository!r}, api_key=<redacted>)"

    def _package_path(self, name: str) -> str:
        base = f"/repos/{quote(self._repository, safe='')}" if self._repository else ""
        return f"{base}/packages/{quote(name, safe='')}"

    def get(
        self, path: str, *, params: dict | None = None, etag: str | None = None
    ) -> tuple[int, Any, str | None]:
        """``(status, json body, etag)``. 304 and 404 are returned, anything else not 200 raises."""
        import requests  # lazy: importing the connectors package loads no client library

        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = self._api_key
        if etag:
            headers["If-None-Match"] = etag
        response = requests.get(BASE_URL + path, params=params, headers=headers, timeout=_TIMEOUT)
        if response.status_code in (304, 404):
            return response.status_code, None, etag
        if response.status_code != 200:
            raise HexpmAPIError(f"Hex.pm {path}: HTTP {response.status_code}")
        return 200, response.json(), response.headers.get("ETag")

    def package(self, name: str, etag: str | None = None) -> tuple[int, Any, str | None]:
        return self.get(self._package_path(name), etag=etag)

    def release(self, name: str, version: str) -> tuple[int, Any, str | None]:
        return self.get(f"{self._package_path(name)}/releases/{quote(version, safe='')}")

    def search(self, query: str) -> list[str]:
        """Names of every package matching ``query``, most recently updated first."""
        names: list[str] = []
        page = 1
        while True:
            _, body, _ = self.get(
                "/packages", params={"search": query, "sort": "updated_at", "page": page}
            )
            if not body:
                return names
            names.extend(item["name"] for item in body)
            page += 1


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def render_package(package: dict) -> dict[str, Any]:
    name = package["name"]
    meta = package.get("meta") or {}
    downloads = package.get("downloads") or {}
    retirements = package.get("retirements") or {}
    releases = sorted(r["version"] for r in package.get("releases") or [])
    lines = [f"Hex package {name}", _text(meta.get("description"))]
    if package.get("repository"):
        lines.append(f"Repository: {package['repository']}")
    if meta.get("licenses"):
        lines.append("Licenses: " + ", ".join(sorted(meta["licenses"])))
    for label, url in sorted((meta.get("links") or {}).items()):
        lines.append(f"Link {label}: {url}")
    if "all" in downloads:
        lines.append(f"Downloads (all time): {downloads['all']}")
    if package.get("latest_stable_version"):
        lines.append(f"Latest stable version: {package['latest_stable_version']}")
    lines.append("Releases: " + ", ".join(releases))
    for version, notice in sorted(retirements.items()):
        lines.append(
            f"Retired release {version}: {notice.get('reason', '')} {_text(notice.get('message'))}"
        )
    return {
        "id": f"package:{name}",
        "title": name,
        "content": "\n".join(line for line in lines if line),
        "url": _text(package.get("html_url")),
        "_deleted": False,
        NODE_SET_COLUMN: [f"hexpm:{name}"],
    }


def render_release(name: str, release: dict) -> dict[str, Any]:
    version = release["version"]
    meta = release.get("meta") or {}
    lines = [f"Hex release {name} {version}"]
    if meta.get("elixir"):
        lines.append(f"Elixir requirement: {meta['elixir']}")
    if meta.get("build_tools"):
        lines.append("Build tools: " + ", ".join(sorted(meta["build_tools"])))
    for dep, req in sorted((release.get("requirements") or {}).items()):
        optional = " (optional)" if req.get("optional") else ""
        lines.append(f"Requires {req.get('app') or dep} {req.get('requirement', '')}{optional}")
    if release.get("has_docs"):
        lines.append(f"Docs: {_text(release.get('docs_html_url'))}")
    if release.get("checksum"):
        lines.append(f"Checksum: {release['checksum']}")
    publisher = (release.get("publisher") or {}).get("username")
    if publisher:
        lines.append(f"Published by: {publisher}")
    notice = release.get("retirement")
    if notice:
        lines.append(f"RETIRED ({notice.get('reason', '')}): {_text(notice.get('message'))}")
    return {
        "id": f"release:{name}:{version}",
        "title": f"{name} {version}",
        "content": "\n".join(lines),
        "url": _text(release.get("html_url")) or _text(release.get("url")),
        "_deleted": False,
        NODE_SET_COLUMN: [f"hexpm:{name}"],
    }


def _forget_package(name: str, known: dict) -> Iterator[dict]:
    yield {"id": f"package:{name}", "_deleted": True}
    for version in known.get("versions", []):
        yield {"id": f"release:{name}:{version}", "_deleted": True}


def _iter_rows(client: Any, state: dict, names: list[str], forget_unselected: bool) -> Iterator:
    packages = state.setdefault("packages", {})
    if forget_unselected:
        for name in sorted(set(packages) - set(names)):
            yield from _forget_package(name, packages.pop(name))
    for name in names:
        known = packages.get(name, {})
        status, package, etag = client.package(name, known.get("etag"))
        if status == 404:
            if name in packages:
                yield from _forget_package(name, packages.pop(name))
            continue
        if status == 304:
            continue
        retirements = package.get("retirements") or {}
        if (
            known
            and known.get("updated_at") == package.get("updated_at")
            and known.get("retirements") == retirements
        ):
            known["etag"] = etag
            continue
        versions = sorted(r["version"] for r in package.get("releases") or [])
        old_retirements = known.get("retirements", {})
        old_versions = set(known.get("versions", []))
        for version in versions:
            if version in old_versions and old_retirements.get(version) == retirements.get(version):
                continue
            _, release, _ = client.release(name, version)
            if release is None:
                raise HexpmAPIError(f"Hex.pm release {name} {version} listed but not found")
            yield render_release(name, release)
        for version in sorted(old_versions - set(versions)):
            yield {"id": f"release:{name}:{version}", "_deleted": True}
        yield render_package(package)
        packages[name] = {
            "etag": etag,
            "updated_at": package.get("updated_at"),
            "retirements": retirements,
            "versions": versions,
        }


def build_hexpm_service(api_key: str | None = None, repository: str | None = None) -> HexpmClient:
    return HexpmClient(api_key, repository)


def hexpm_source(
    resource_name: str,
    package_names: list[str] | None = None,
    search: str | None = None,
    api_key: str | None = None,
    repository: str | None = None,
    service: Any = None,
):
    """A ``dlt`` resource of Hex.pm package and release documents.

    Args:
        resource_name: Stable name of this connection. It names the staging table and scopes the
            sync state. There is no default.
        package_names: Hex package names to sync. Packages that leave this list are forgotten.
        search: A Hex.pm search query; its matches are synced, and never forgotten for leaving it.
        api_key: Optional read-only key for private organizations (with ``repository``).
        repository: Hex organization whose private packages are read.
        service: Pre-built client (see :func:`build_hexpm_service`); an injection point for tests.

    Returns:
        A ``dlt`` resource with ``primary_key="id"``, ``write_disposition="merge"`` and a
        ``_deleted`` hard-delete column.
    """
    import dlt

    if not resource_name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", resource_name):
        raise ValueError("resource_name must be a non-empty identifier")
    names = sorted(set(package_names or []))
    if not names and not search:
        raise ValueError("hexpm_source needs package_names or search")
    for name in names:
        if not _NAME.match(name):
            raise ValueError(f"Invalid Hex package name: {name!r}")
    client = service if service is not None else HexpmClient(api_key, repository)

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def hexpm_documents():
        selected = sorted(set(names) | set(client.search(search) if search else []))
        yield from _iter_rows(
            client, dlt.current.resource_state(), selected, forget_unselected=not search
        )

    resource = hexpm_documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "hexpm")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    return resource
