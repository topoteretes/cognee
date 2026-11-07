"""Packagist / Composer connector for cognee: a ``dlt`` source over the PHP package registry.

One source is one selection of packages. It yields a document per package (maintainers,
repository, ``abandoned`` replacement notice, and every release with its ``require``,
``provide`` and ``conflict`` constraints) and a document per security advisory, and is meant
to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import packagist_source

    await cognee.remember(
        packagist_source(
            packages=["laravel/framework", "monolog/monolog"],
            user_agent="my-team-cognee (ops@example.com)",
        ),
        dataset_name="packagist",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Selection** is exactly one of ``packages`` (``vendor/package`` names), ``vendor`` (every
  package of that vendor, from ``/packages/list.json?vendor=``) or ``composer_file`` (the
  ``require``/``require-dev`` names of a ``composer.json``, or the ``packages`` and
  ``packages-dev`` of a ``composer.lock``; ``php`` and ``ext-*`` platform entries are skipped).
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are
  ``package:<vendor/name>`` and ``advisory:<PKSA id>``. Every column feeds the stored
  document identity, so nothing volatile is rendered: download counts and favers live
  nowhere in the text. An advisory's text names its package by the same ``vendor/name`` as
  the package document's title, so the graph can join them.
* **Node set** ``packagist:packages`` / ``packagist:advisories``.
* **Incremental** by Packagist's ``/metadata/changes.json`` feed. The first run reads the
  feed's current timestamp (10,000ths of a second) and fetches every selected package; later
  runs re-fetch only selected packages with an ``update`` action since the saved timestamp,
  and tombstone those with a ``delete`` action. A ``resync`` action (the cursor is older than
  the feed keeps) re-fetches everything. The cursor lives in dlt resource state and is
  advanced only after the run's rows were produced.
* **Advisories** have no change feed: they are read for the re-fetched packages and a row is
  emitted when its rendered text changed. An advisory that a package no longer lists is
  tombstoned.
* **Auth** is none for public Packagist. ``user_agent`` is mandatory (Packagist's API
  policy); ``token`` is an optional Bearer token for Private Packagist or a self-hosted
  instance, which also needs ``repo_url`` and ``api_url``.
"""

import hashlib
import json
import re
from collections.abc import Iterator
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

PACKAGES_NODE_SET = "packagist:packages"
ADVISORIES_NODE_SET = "packagist:advisories"
_TIMEOUT = 30
_PACKAGE_NAME = re.compile(r"^[a-z0-9]([_.-]?[a-z0-9]+)*/[a-z0-9]([_.-]?[a-z0-9]+)*$")
# Advisories are requested in chunks to keep the form body under server limits.
_ADVISORY_CHUNK = 100


class PackagistAPIError(RuntimeError):
    """A Packagist request failed. Carries the status, never a response body."""


class PackagistClient:
    """Minimal synchronous client. Tests and hosts may pass any object with its methods."""

    def __init__(
        self,
        user_agent: str,
        *,
        token: str | None = None,
        repo_url: str = "https://repo.packagist.org",
        api_url: str = "https://packagist.org",
    ):
        if not user_agent or not user_agent.strip():
            raise ValueError("Packagist requires a descriptive user_agent")
        self._headers = {"User-Agent": user_agent.strip()}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._repo_url = repo_url.rstrip("/")
        self._api_url = api_url.rstrip("/")

    def __repr__(self) -> str:
        return "PackagistClient(token=<redacted>)"

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        import requests

        try:
            response = requests.request(
                method,
                url,
                headers=self._headers,
                timeout=_TIMEOUT,
                allow_redirects=False,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise PackagistAPIError(f"Packagist request failed: {type(exc).__name__}") from None
        return response

    def _json(self, method: str, url: str, *, missing_ok: bool = False, **kwargs: Any) -> Any:
        response = self._request(method, url, **kwargs)
        if response.status_code == 404 and missing_ok:
            return None
        if response.status_code != 200:
            raise PackagistAPIError(f"Packagist request failed: HTTP {response.status_code}")
        return response.json()

    def list_vendor(self, vendor: str) -> list[str]:
        body = self._json("GET", f"{self._api_url}/packages/list.json", params={"vendor": vendor})
        return list(body["packageNames"])

    def changes(self, since: int | None) -> dict:
        """The change feed. Without ``since`` Packagist answers 400 with the current timestamp."""
        if since is not None:
            return self._json(
                "GET", f"{self._api_url}/metadata/changes.json", params={"since": since}
            )
        response = self._request("GET", f"{self._api_url}/metadata/changes.json")
        body = response.json()
        if "timestamp" not in body:
            raise PackagistAPIError(f"Packagist request failed: HTTP {response.status_code}")
        return body

    def metadata(self, name: str) -> dict | None:
        """Composer v2 release metadata (minified), or ``None`` when the package is gone."""
        return self._json("GET", f"{self._repo_url}/p2/{name}.json", missing_ok=True)

    def package(self, name: str) -> dict | None:
        """The package page JSON, or ``None`` when the package is gone."""
        body = self._json("GET", f"{self._api_url}/packages/{name}.json", missing_ok=True)
        return None if body is None else body["package"]

    def advisories(self, names: list[str]) -> dict[str, list[dict]]:
        body = self._json(
            "POST",
            f"{self._api_url}/api/security-advisories/",
            data={"packages[]": names},
        )
        return body["advisories"]


def build_packagist_service(user_agent: str, token: str | None = None, **urls: str):
    """Build the HTTP client for :func:`packagist_source`."""
    return PackagistClient(user_agent, token=token, **urls)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def composer_file_packages(path: str) -> list[str]:
    """Package names listed by a ``composer.json`` or ``composer.lock``."""
    with open(path, encoding="utf-8") as handle:
        body = json.load(handle)
    names: set[str] = set()
    if "packages" in body or "packages-dev" in body:
        names.update(
            entry["name"] for key in ("packages", "packages-dev") for entry in body.get(key, [])
        )
    else:
        for key in ("require", "require-dev"):
            names.update(body.get(key, {}))
    return sorted(name for name in names if "/" in name)


def _selected(client: Any, packages, vendor, composer_file) -> list[str]:
    if sum(x is not None for x in (packages, vendor, composer_file)) != 1:
        raise ValueError("packagist_source needs exactly one of packages, vendor, composer_file")
    if packages is not None:
        names = list(packages)
    elif vendor is not None:
        names = client.list_vendor(vendor)
    else:
        names = composer_file_packages(composer_file)
    names = sorted({name.lower() for name in names})
    for name in names:
        if not _PACKAGE_NAME.match(name):
            raise ValueError(f"Not a vendor/package name: {name!r}")
    return names


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def expand_versions(metadata: dict, name: str) -> list[dict]:
    """Undo Composer v2 minification: each version only lists keys that changed."""
    expanded: list[dict] = []
    previous: dict = {}
    for entry in metadata["packages"][name]:
        current = dict(previous)
        for key, value in entry.items():
            if value == "__unset":
                current.pop(key, None)
            else:
                current[key] = value
        expanded.append(current)
        previous = current
    return expanded


def _constraints(label: str, mapping: dict | None) -> list[str]:
    return [f"  {label} {dep} {mapping[dep]}" for dep in sorted(mapping or {})]


def render_package(name: str, package: dict, versions: list[dict]) -> dict[str, Any]:
    lines = [f"Packagist package {name}."]
    if package.get("description"):
        lines.append(f"Description: {package['description']}")
    lines.append(f"Repository: {package.get('repository', '')}")
    maintainers = sorted(m["name"] for m in package.get("maintainers", []))
    if maintainers:
        lines.append("Maintainers: " + ", ".join(maintainers))
    abandoned = package.get("abandoned")
    if abandoned:
        replacement = abandoned if isinstance(abandoned, str) else None
        lines.append(
            f"Abandoned: yes, replaced by {replacement}"
            if replacement
            else "Abandoned: yes, no replacement recommended"
        )
    else:
        lines.append("Abandoned: no")
    lines.append("Releases:")
    for version in versions:
        lines.append(f"- {version['version']} (released {version.get('time', '')})")
        php = (version.get("require") or {}).get("php")
        if php:
            lines.append(f"  PHP {php}")
        lines += _constraints(
            "requires", {k: v for k, v in (version.get("require") or {}).items() if k != "php"}
        )
        lines += _constraints("provides", version.get("provide"))
        lines += _constraints("conflicts with", version.get("conflict"))
    return {
        "id": f"package:{name}",
        "title": name,
        "content": "\n".join(lines),
        "url": f"https://packagist.org/packages/{name}",
        "_deleted": False,
        "cognee_node_set": [PACKAGES_NODE_SET],
    }


def render_advisory(name: str, advisory: dict) -> dict[str, Any]:
    lines = [
        f"Packagist security advisory {advisory['advisoryId']} for package {name}.",
        f"Title: {advisory.get('title', '')}",
    ]
    if advisory.get("cve"):
        lines.append(f"CVE: {advisory['cve']}")
    lines.append(f"Affected versions: {advisory.get('affectedVersions', '')}")
    if advisory.get("link"):
        lines.append(f"Link: {advisory['link']}")
    return {
        "id": f"advisory:{advisory['advisoryId']}",
        "title": f"{advisory['advisoryId']}: {advisory.get('title', '')}",
        "content": "\n".join(lines),
        "url": advisory.get("link") or f"https://packagist.org/packages/{name}",
        "_deleted": False,
        "cognee_node_set": [ADVISORIES_NODE_SET],
    }


def _tombstone(row_id: str) -> dict[str, Any]:
    return {"id": row_id, "title": "", "content": "", "url": "", "_deleted": True}


def _fingerprint(row: dict) -> str:
    return hashlib.sha256(row["content"].encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------
def _changed_packages(
    client: Any, selected: list[str], state: dict
) -> tuple[set[str], set[str], int]:
    """Return (packages to fetch, packages deleted, new cursor)."""
    since = state.get("since")
    if since is None:
        return set(selected), set(), client.changes(None)["timestamp"]
    feed = client.changes(since)
    wanted = set(selected)
    fetch: set[str] = set()
    deleted: set[str] = set()
    for action in feed["actions"]:
        if action["type"] == "resync":
            return wanted, set(), feed["timestamp"]
        # Dev branches are listed as ``vendor/package~dev``.
        package = action["package"].split("~")[0].lower()
        if package not in wanted:
            continue
        if action["type"] == "update":
            fetch.add(package)
        elif action["type"] == "delete":
            deleted.add(package)
    return fetch - deleted, deleted, feed["timestamp"]


def _iter_rows(
    client: Any, selected: list[str], state: dict, stats: dict[str, int]
) -> Iterator[dict]:
    """Yield the changed rows. Pure of dlt, so tests drive it with a dict."""
    fetch, deleted, cursor = _changed_packages(client, selected, state)
    package_advisories: dict[str, list[str]] = state.setdefault("advisory_ids", {})
    fingerprints: dict[str, str] = state.setdefault("fingerprints", {})

    def forget(name: str) -> Iterator[dict]:
        for row_id in [
            f"package:{name}",
            *(f"advisory:{a}" for a in package_advisories.pop(name, [])),
        ]:
            if fingerprints.pop(row_id, None) is not None:
                stats["deleted"] += 1
                yield _tombstone(row_id)

    def emit(row: dict) -> Iterator[dict]:
        stats["scanned"] += 1
        fingerprint = _fingerprint(row)
        if fingerprints.get(row["id"]) == fingerprint:
            stats["skipped"] += 1
            return
        fingerprints[row["id"]] = fingerprint
        stats["synced"] += 1
        yield row

    for name in sorted(deleted):
        yield from forget(name)

    live = []
    for name in sorted(fetch):
        metadata = client.metadata(name)
        package = client.package(name) if metadata is not None else None
        if metadata is None or package is None:
            yield from forget(name)
            continue
        live.append(name)
        yield from emit(render_package(name, package, expand_versions(metadata, name)))

    for start in range(0, len(live), _ADVISORY_CHUNK):
        chunk = live[start : start + _ADVISORY_CHUNK]
        found = client.advisories(chunk)
        for name in chunk:
            advisories = sorted(found.get(name, []), key=lambda a: a["advisoryId"])
            ids = [a["advisoryId"] for a in advisories]
            for gone in set(package_advisories.get(name, [])) - set(ids):
                if fingerprints.pop(f"advisory:{gone}", None) is not None:
                    stats["deleted"] += 1
                    yield _tombstone(f"advisory:{gone}")
            package_advisories[name] = ids
            for advisory in advisories:
                yield from emit(render_advisory(name, advisory))

    state["since"] = cursor


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def packagist_source(
    *,
    user_agent: str,
    packages: list[str] | None = None,
    vendor: str | None = None,
    composer_file: str | None = None,
    resource_name: str = "packagist_packages",
    service: Any = None,
    token: str | None = None,
    repo_url: str = "https://repo.packagist.org",
    api_url: str = "https://packagist.org",
):
    """Return a ``dlt`` resource that yields Packagist package and advisory documents.

    Args:
        user_agent: Descriptive ``User-Agent`` Packagist's API policy requires.
        packages: ``vendor/package`` names. Exactly one selection argument is required.
        vendor: Every package of this vendor.
        composer_file: Path of a ``composer.json`` or ``composer.lock``.
        resource_name: Stable, connection-specific table name.
        service: Pre-built client (see :func:`build_packagist_service`), an injection point
            for tests and hosts.
        token: Optional Bearer token for Private Packagist / self-hosted instances.
        repo_url: Base URL of the Composer v2 metadata host.
        api_url: Base URL of the Packagist API host.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Packagist sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    client = service or build_packagist_service(
        user_agent, token, repo_url=repo_url, api_url=api_url
    )
    selected = _selected(client, packages, vendor, composer_file)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def packagist_packages():
        stats.clear()
        stats.update(scanned=0, skipped=0, synced=0, deleted=0, failed=0)
        yield from _iter_rows(client, selected, dlt.current.resource_state(), stats)

    resource = packagist_packages()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "packagist")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
