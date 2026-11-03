"""crates.io connector for cognee: a ``dlt`` source that yields one document per Rust crate.

Each crate is read from the public crates.io API (no key, only a descriptive ``User-Agent``,
which crates.io requires) and written as one document: description, latest stable (or the
requested) version, documentation and repository links, license, keywords, feature flags,
dependencies and the README. Only that one version is read, never the release history.

A run is a full snapshot (``write_disposition="replace"``): the document id is the hash of the
document text, so a crate whose registry data did not change keeps its id.
"""

import hashlib
import time
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

logger = get_logger("crates_connector")

API_URL = "https://crates.io/api/v1"
DEFAULT_USER_AGENT = "cognee-crates-connector (https://github.com/topoteretes/cognee)"
_DEPENDENCY_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


class CratesSourceError(RuntimeError):
    """A crates.io request failed or a ``Cargo.toml`` could not be used."""


class _TextExtractor(HTMLParser):
    """Turns the rendered README HTML crates.io serves into plain text."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _readme_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return "".join(extractor.parts).strip()


def _dependency_names(cargo_toml: Path) -> list[str]:
    """Registry crate names a ``Cargo.toml`` depends on, sorted, renamed packages resolved.

    Reads ``[dependencies]``, ``[dev-dependencies]``, ``[build-dependencies]``,
    ``[workspace.dependencies]`` and their ``[target.*]`` variants. Path and git dependencies
    are not on crates.io and are skipped.
    """
    import tomllib

    try:
        manifest = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CratesSourceError(f"Could not read {cargo_toml.name}: {error}") from error

    tables = [manifest.get(name, {}) for name in _DEPENDENCY_TABLES]
    tables.append(manifest.get("workspace", {}).get("dependencies", {}))
    for target in manifest.get("target", {}).values():
        tables.extend(target.get(name, {}) for name in _DEPENDENCY_TABLES)

    names = set()
    for table in tables:
        for key, spec in table.items():
            if isinstance(spec, dict):
                if "path" in spec or "git" in spec:
                    continue
                names.add(spec.get("package", key))
            else:
                names.add(key)
    return sorted(names)


def _parse_crate_spec(spec: str) -> tuple[str, str | None]:
    """``"tokio"`` or ``"tokio@1.38.0"`` to ``(name, version or None)``."""
    name, _, version = spec.partition("@")
    if not name.strip():
        raise ValueError(f"Invalid crate name {spec!r}.")
    return name.strip(), version.strip() or None


def _get(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url, follow_redirects=True)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise CratesSourceError(f"crates.io returned {response.status_code} for {url}") from error
    return response


def _crate_document(client: httpx.Client, name: str, version: str | None) -> dict[str, Any]:
    """Fetch one crate and build its document row."""
    payload = _get(client, f"{API_URL}/crates/{name}").json()
    crate = payload["crate"]
    version = version or crate.get("max_stable_version") or crate["max_version"]
    release = next((item for item in payload["versions"] if item["num"] == version), None)
    if release is None:
        raise CratesSourceError(f"Crate {name!r} has no version {version!r}.")

    dependencies = _get(client, f"{API_URL}/crates/{name}/{version}/dependencies").json()
    readme = _readme_text(_get(client, f"{API_URL}/crates/{name}/{version}/readme").text)

    lines = [f"# {name} {version}", "", crate.get("description") or ""]
    lines += [
        f"License: {release.get('license') or 'unknown'}",
        f"Documentation: {crate.get('documentation') or f'https://docs.rs/{name}/{version}'}",
        f"Repository: {crate.get('repository') or ''}",
        f"Homepage: {crate.get('homepage') or ''}",
        f"Keywords: {', '.join(crate.get('keywords') or [])}",
        "",
        "## Features",
    ]
    features = release.get("features") or {}
    lines += [f"- {feature}: {', '.join(enabled)}" for feature, enabled in sorted(features.items())]
    lines += ["", "## Dependencies"]
    for dependency in sorted(
        dependencies["dependencies"], key=lambda item: (item["kind"], item["crate_id"])
    ):
        flags = [dependency["kind"]]
        if dependency.get("optional"):
            flags.append("optional")
        if dependency.get("features"):
            flags.append(f"features: {', '.join(dependency['features'])}")
        lines.append(f"- {dependency['crate_id']} {dependency['req']} ({'; '.join(flags)})")
    lines += ["", "## README", readme]

    content = "\n".join(lines)
    return {
        "id": hashlib.sha256(content.encode()).hexdigest(),
        "title": f"{name} {version}",
        "content": content,
        "url": f"https://crates.io/crates/{name}/{version}",
    }


def _iter_rows(
    specs: list[tuple[str, str | None]], client: httpx.Client, request_interval: float
) -> Iterator[dict[str, Any]]:
    for index, (name, version) in enumerate(specs):
        if index and request_interval:
            time.sleep(request_interval)
        yield _crate_document(client, name, version)
    logger.info("crates.io: %d crate(s) read.", len(specs))


def crates_source(
    crate_names: list[str] | tuple[str, ...] | None = None,
    cargo_toml: str | Path | None = None,
    resource_name: str = "crates",
    user_agent: str = DEFAULT_USER_AGENT,
    request_interval: float = 1.0,
    client: httpx.Client | None = None,
):
    """Return a ``dlt`` resource that yields one document per crate.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="replace"``.

    Args:
        crate_names: Crates to read, ``"tokio"`` for the latest stable release or
            ``"tokio@1.38.0"`` for a specific one.
        cargo_toml: Path of a ``Cargo.toml`` whose registry dependencies are read (latest
            stable release of each). Combined with ``crate_names`` when both are given.
        resource_name: Names the staging table.
        user_agent: Identifies the caller to crates.io, which rejects anonymous requests.
        request_interval: Seconds between crates; crates.io's crawler policy
            allows at most one request per second.
        client: An ``httpx.Client`` to use instead of building one (tests, proxies).
    """
    import dlt

    if not crate_names and cargo_toml is None:
        raise ValueError("Pass crate_names, a cargo_toml path, or both.")
    if isinstance(crate_names, str):
        raise ValueError('crate_names must be a list like ["tokio", "serde"].')
    if not user_agent.strip():
        raise ValueError(
            "user_agent must be a non-empty string; crates.io rejects requests without one."
        )

    specs = dict(_parse_crate_spec(spec) for spec in crate_names or [])
    if cargo_toml is not None:
        path = Path(cargo_toml).expanduser()
        if not path.is_file():
            raise ValueError("cargo_toml must be an existing file.")
        for name in _dependency_names(path):
            specs.setdefault(name, None)
    ordered = sorted(specs.items())

    @dlt.resource(name=resource_name, primary_key="id", write_disposition="replace")
    def crates():
        http = client or httpx.Client(headers={"User-Agent": user_agent}, timeout=30)
        try:
            yield from dlt_utils.guarded_rows(_iter_rows(ordered, http, request_interval))
        finally:
            if client is None:
                http.close()

    resource = crates()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "crates")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    return resource
