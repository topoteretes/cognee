"""NuGet v3 connector for cognee: a ``dlt`` source that turns .NET packages into memory.

One source is one NuGet v3 feed (``https://api.nuget.org/v3/index.json`` by default, or a
private feed). It yields one document per package and one per listed version, rendered from the
feed's registration index: description, authors, license, target-framework dependency groups,
deprecation notices (with the recommended alternate package) and known vulnerabilities::

    import cognee
    from cognee.tasks.ingestion.connectors import nuget_source

    await cognee.remember(
        nuget_source(resource_name="nuget_deps", package_ids=["Newtonsoft.Json"]),
        dataset_name="nuget",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
    )

Design
------
* **Service discovery.** ``RegistrationsBaseUrl/Versioned`` and ``SearchQueryService`` are read
  from the service index, no CDN host is hardcoded.
* **Selection** is exactly one of ``package_ids``, ``query`` (search service) or ``project_file``
  (``packages.lock.json``, ``.csproj`` or ``Directory.Packages.props``).
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``; ids are
  ``nuget:<id-lower>`` and ``nuget:<id-lower>@<version>``. Node set ``nuget:<id-lower>``.
* **Incremental.** Each registration page's ``commitId`` is kept in resource state together
  with the versions it holds: an unchanged package (same index ``commitId``) costs one request
  and an unchanged page emits nothing.
* **Forget-on-delete.** A version that left its page, a package whose registration answers 404,
  and a package that left the selection are emitted as ``_deleted`` tombstones. Any other
  error raises and forgets nothing.
"""

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

logger = get_logger("nuget_connector")

DEFAULT_SERVICE_INDEX = "https://api.nuget.org/v3/index.json"
REGISTRATION_TYPE = "RegistrationsBaseUrl/Versioned"
SEARCH_TYPE = "SearchQueryService"
_SEARCH_PAGE = 100
_PROJECT_FILES = ("packages.lock.json", ".csproj", "Directory.Packages.props")


class NuGetSourceError(RuntimeError):
    """Base error of the NuGet source."""


def _package_key(package_id: str) -> str:
    return f"nuget:{package_id.lower()}"


def _version_key(package_id: str, version: str) -> str:
    return f"nuget:{package_id.lower()}@{version.lower()}"


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# ---------------------------------------------------------------- HTTP / discovery


def build_nuget_client(
    *,
    api_key: str | None = None,
    basic_auth: tuple[str, str] | None = None,
    bearer_token: str | None = None,
):
    """Build the ``httpx.Client`` for a feed. Auth is optional; at most one kind is allowed."""
    import httpx

    if sum(x is not None for x in (api_key, basic_auth, bearer_token)) > 1:
        raise ValueError("Pass at most one of api_key, basic_auth and bearer_token.")
    headers = {}
    auth = None
    if api_key is not None:
        headers["X-NuGet-ApiKey"] = api_key
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if basic_auth is not None:
        auth = httpx.BasicAuth(*basic_auth)
    return httpx.Client(headers=headers, auth=auth, timeout=30.0, follow_redirects=True)


def _get_json(client, url: str) -> dict:
    # httpx decodes the gzip of ``registration5-gz-semver2`` transparently.
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def _discover(client, service_index_url: str) -> dict[str, str]:
    """The registration and search base urls of the feed, from its service index."""
    resources = _get_json(client, service_index_url).get("resources", [])
    found: dict[str, str] = {}
    for resource in resources:
        for kind in (REGISTRATION_TYPE, SEARCH_TYPE):
            if kind in _as_list(resource.get("@type")):
                found.setdefault(kind, resource["@id"])
    if REGISTRATION_TYPE not in found:
        raise NuGetSourceError(f"The feed has no {REGISTRATION_TYPE} service.")
    return found


# ---------------------------------------------------------------- selection


def _ids_from_project_file(path: Path) -> list[str]:
    name = path.name
    if name == "packages.lock.json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            package_id
            for framework in data.get("dependencies", {}).values()
            for package_id in framework
        ]
    if name.endswith(".csproj") or name == "Directory.Packages.props":
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        tags = ("PackageReference", "PackageVersion")
        return [
            element.get("Include") or element.get("Update") or ""
            for element in root.iter()
            if element.tag.rpartition("}")[2] in tags
        ]
    raise ValueError(f"project_file must be one of {_PROJECT_FILES}, got {name!r}.")


def _clean_ids(ids: list[str]) -> list[str]:
    cleaned = {}
    for package_id in ids:
        if not isinstance(package_id, str) or not package_id.strip():
            raise ValueError("Every package id must be a non-empty string.")
        cleaned.setdefault(package_id.strip().lower(), package_id.strip())
    return [cleaned[key] for key in sorted(cleaned)]


def _search_ids(client, search_url: str, query: str) -> list[str]:
    ids: list[str] = []
    while True:
        data = _get_json(
            client, f"{search_url}?q={query}&take={_SEARCH_PAGE}&skip={len(ids)}&semVerLevel=2.0.0"
        )
        page = [entry["id"] for entry in data.get("data", [])]
        ids.extend(page)
        if not page or len(ids) >= data.get("totalHits", 0):
            return ids


# ---------------------------------------------------------------- rendering


def _deprecation_lines(deprecation: dict | None) -> list[str]:
    if not deprecation:
        return []
    reasons = ", ".join(sorted(_as_list(deprecation.get("reasons")))) or "unspecified"
    lines = [f"Deprecated. Reasons: {reasons}."]
    if deprecation.get("message"):
        lines.append(f"Deprecation message: {deprecation['message']}")
    alternate = deprecation.get("alternatePackage")
    if alternate:
        lines.append(
            f"Recommended alternate package: {alternate.get('id')} "
            f"(version range {alternate.get('range') or '*'})."
        )
    return lines


def _dependency_lines(groups: list[dict] | None) -> list[str]:
    lines = []
    for group in sorted(groups or [], key=lambda g: g.get("targetFramework") or ""):
        framework = group.get("targetFramework") or "any framework"
        deps = sorted(
            f"{d.get('id')} {d.get('range') or '*'}" for d in group.get("dependencies", [])
        )
        lines.append(f"Target framework {framework} depends on: {', '.join(deps) or 'nothing'}.")
    return lines


def _vulnerability_lines(vulnerabilities: list[dict] | None) -> list[str]:
    return sorted(
        f"Vulnerability advisory {v.get('advisoryUrl')} (severity {v.get('severity')})."
        for v in vulnerabilities or []
    )


def _entry_text(package_id: str, entry: dict) -> str:
    version = entry.get("version", "")
    lines = [f"NuGet package {package_id} version {version}."]
    if entry.get("description"):
        lines.append(str(entry["description"]).strip())
    authors = entry.get("authors")
    if authors:
        lines.append(f"Authors: {authors if isinstance(authors, str) else ', '.join(authors)}.")
    license_text = entry.get("licenseExpression") or entry.get("licenseUrl")
    if license_text:
        lines.append(f"License: {license_text}.")
    lines.append(f"Listed: {'yes' if entry.get('listed', True) else 'no'}.")
    lines += _dependency_lines(entry.get("dependencyGroups"))
    lines += _deprecation_lines(entry.get("deprecation"))
    lines += _vulnerability_lines(entry.get("vulnerabilities"))
    return "\n".join(lines)


def _version_row(package_id: str, entry: dict) -> dict[str, Any]:
    version = entry["version"]
    return {
        "id": _version_key(package_id, version),
        "title": f"{package_id} {version}",
        "content": _entry_text(package_id, entry),
        "url": f"https://www.nuget.org/packages/{package_id}/{version}",
        "_deleted": False,
        NODE_SET_COLUMN: [_package_key(package_id)],
    }


def _package_row(package_id: str, latest: dict, versions: list[str]) -> dict[str, Any]:
    lines = [_entry_text(package_id, latest).replace(" version " + latest["version"], "", 1)]
    lines.append(f"Latest version: {latest['version']}. Versions: {', '.join(versions)}.")
    return {
        "id": _package_key(package_id),
        "title": package_id,
        "content": "\n".join(lines),
        "url": f"https://www.nuget.org/packages/{package_id}",
        "_deleted": False,
        NODE_SET_COLUMN: [_package_key(package_id)],
    }


# ---------------------------------------------------------------- sync


def _version_sort_key(version: str) -> tuple:
    main, _, pre = version.split("+")[0].partition("-")
    numbers = tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.]", main))
    # A release sorts after its prereleases.
    return (numbers, pre == "", pre)


def _load_pages(client, index: dict) -> Iterator[tuple[str, str, list[dict]]]:
    """Yield ``(page url, page commitId, leaf catalog entries)`` for each registration page."""
    for page in index.get("items", []):
        items = page.get("items")
        if items is None:
            page_data = _get_json(client, page["@id"])
            items = page_data.get("items", [])
        yield page["@id"], page.get("commitId", ""), [leaf["catalogEntry"] for leaf in items]


def _sync_package(client, registration_base: str, package_id: str, pstate: dict, stats: dict):
    """Yield rows for one package and update ``pstate``; a 404 tombstones the package."""
    import httpx

    url = f"{registration_base.rstrip('/')}/{package_id.lower()}/index.json"
    try:
        index = _get_json(client, url)
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 404:
            raise
        yield from _tombstones(pstate)
        stats["deleted"] += 1 if pstate else 0
        pstate.clear()
        return
    if pstate.get("commit") and pstate["commit"] == index.get("commitId"):
        stats["unchanged"] += 1
        return

    old_pages = pstate.get("pages", {})
    new_pages: dict[str, dict] = {}
    latest: dict | None = None
    changed = False
    for page_url, commit, entries in _load_pages(client, index):
        for entry in entries:
            if latest is None or _version_sort_key(entry["version"]) >= _version_sort_key(
                latest["version"]
            ):
                latest = entry
        previous = old_pages.get(page_url)
        listed = [e for e in entries if e.get("listed", True)]
        new_pages[page_url] = {"commit": commit, "versions": sorted(e["version"] for e in listed)}
        if previous and previous["commit"] == commit and commit:
            continue
        changed = True
        for entry in listed:
            yield _version_row(package_id, entry)
            stats["synced"] += 1
        kept = {e["version"] for e in listed}
        for version in (previous or {}).get("versions", []):
            if version not in kept:
                yield {"id": _version_key(package_id, version), "_deleted": True}
                stats["deleted"] += 1
    for page_url in set(old_pages) - set(new_pages):
        changed = True
        for version in old_pages[page_url]["versions"]:
            yield {"id": _version_key(package_id, version), "_deleted": True}
            stats["deleted"] += 1
    if latest is not None and (changed or not pstate):
        all_versions = sorted(
            {v for page in new_pages.values() for v in page["versions"]}, key=_version_sort_key
        )
        yield _package_row(package_id, latest, all_versions)
        stats["synced"] += 1
    pstate.update(commit=index.get("commitId", ""), pages=new_pages, name=package_id)


def _tombstones(pstate: dict) -> Iterator[dict]:
    for page in pstate.get("pages", {}).values():
        for version in page["versions"]:
            yield {"id": _version_key(pstate["name"], version), "_deleted": True}
    if pstate.get("name"):
        yield {"id": _package_key(pstate["name"]), "_deleted": True}


def _iter_rows(
    client,
    service_index_url: str,
    package_ids: list[str] | None,
    query: str | None,
    project_file: Path | None,
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the rows of every selected package, then tombstones for packages that left."""
    services = _discover(client, service_index_url)
    if package_ids is not None:
        selected = package_ids
    elif project_file is not None:
        selected = _clean_ids(_ids_from_project_file(project_file))
    else:
        if SEARCH_TYPE not in services:
            raise NuGetSourceError(f"The feed has no {SEARCH_TYPE} service.")
        selected = _clean_ids(_search_ids(client, services[SEARCH_TYPE], query or ""))
    packages = state.setdefault("packages", {})
    for package_id in selected:
        if check_active is not None:
            check_active()
        pstate = packages.setdefault(package_id.lower(), {})
        yield from _sync_package(client, services[REGISTRATION_TYPE], package_id, pstate, stats)
        if not pstate:
            del packages[package_id.lower()]
    wanted = {p.lower() for p in selected}
    for key in sorted(set(packages) - wanted):
        yield from _tombstones(packages[key])
        stats["deleted"] += 1
        del packages[key]


def nuget_source(
    *,
    resource_name: str,
    package_ids: list[str] | None = None,
    query: str | None = None,
    project_file: str | Path | None = None,
    service_index_url: str = DEFAULT_SERVICE_INDEX,
    api_key: str | None = None,
    basic_auth: tuple[str, str] | None = None,
    bearer_token: str | None = None,
    client: Any = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields NuGet package and version documents.

    Args:
        resource_name: Stable name of this connection; scopes the sync state and the cleanup
            of deleted records. Keep it fixed across runs.
        package_ids: Package ids to sync.
        query: Search query (``SearchQueryService``) whose hits are synced.
        project_file: ``packages.lock.json``, ``*.csproj`` or ``Directory.Packages.props``.
        service_index_url: The v3 ``index.json`` of the feed.
        api_key, basic_auth, bearer_token: Optional auth for private feeds (at most one).
        client: Pre-built ``httpx.Client``; an injection point for tests and hosts.
        check_active: Optional host authorization checkpoint, called before each package.

    Exactly one of ``package_ids``, ``query`` and ``project_file`` must be given.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 2:
        raise RuntimeError(
            "NuGet sync requires a Cognee build that reads per-row node sets "
            "(DOCUMENT_SYNC_VERSION >= 2). Upgrade Cognee before syncing."
        )
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("nuget_source needs a non-empty resource_name.")
    if sum(x is not None for x in (package_ids, query, project_file)) != 1:
        raise ValueError("Pass exactly one of package_ids, query and project_file.")
    if package_ids is not None:
        if not isinstance(package_ids, (list, tuple)) or not package_ids:
            raise ValueError("package_ids must be a non-empty list.")
        package_ids = _clean_ids(list(package_ids))
    project_path = Path(project_file).expanduser() if project_file is not None else None
    if project_path is not None and not project_path.is_file():
        raise ValueError("project_file must be an existing file.")
    http = (
        client
        if client is not None
        else build_nuget_client(api_key=api_key, basic_auth=basic_auth, bearer_token=bearer_token)
    )
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def nuget_packages():
        stats.update(synced=0, deleted=0, unchanged=0)
        rows = _iter_rows(
            http,
            service_index_url,
            package_ids,
            query,
            project_path,
            dlt.current.resource_state(),
            stats,
            check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = nuget_packages()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "nuget")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
