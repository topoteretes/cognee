"""DLT source for the NuGet v3 package registry API."""

import typing
from collections.abc import Iterable
import httpx

try:
    import dlt
except ImportError:
    # dlt is optional or provided by host runtime
    dlt = None  # type: ignore[assignment]


def _fetch_package_data(client: httpx.Client, package_name: str) -> dict[str, typing.Any] | None:
    pkg_id = package_name.strip().lower()
    if not pkg_id:
        return None

    url = f"https://api.nuget.org/v3/registration5-gz-semver2/{pkg_id}/index.json"
    response = client.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    data = response.json()

    versions: list[dict[str, typing.Any]] = []
    latest_deprecation = None
    all_vulnerabilities: list[dict[str, typing.Any]] = []

    pages = data.get("items", [])
    for page in pages:
        items = page.get("items")
        if items is None:
            page_id = page.get("@id")
            if page_id:
                page_resp = client.get(page_id)
                if page_resp.status_code == 200:
                    items = page_resp.json().get("items", [])
                else:
                    items = []
            else:
                items = []

        for item in (items or []):
            entry = item.get("catalogEntry", {})
            version_str = entry.get("version")
            dep_groups = entry.get("dependencyGroups", [])
            deprecation = entry.get("deprecation")
            vulnerabilities = entry.get("vulnerabilities", [])

            if deprecation:
                latest_deprecation = deprecation
            if vulnerabilities:
                all_vulnerabilities.extend(vulnerabilities)

            versions.append(
                {
                    "version": version_str,
                    "published": entry.get("published"),
                    "authors": entry.get("authors"),
                    "description": entry.get("description"),
                    "dependency_groups": dep_groups,
                    "deprecation": deprecation,
                    "vulnerabilities": vulnerabilities,
                }
            )

    latest_entry = versions[-1] if versions else {}

    return {
        "package_id": package_name,
        "id": pkg_id,
        "latest_version": latest_entry.get("version"),
        "authors": latest_entry.get("authors"),
        "description": latest_entry.get("description"),
        "deprecation": latest_deprecation,
        "vulnerabilities": all_vulnerabilities,
        "versions": versions,
    }


def _nuget_resource_generator(package_names: Iterable[str]) -> Iterable[dict[str, typing.Any]]:
    with httpx.Client(timeout=30.0) as client:
        for pkg_name in package_names:
            data = _fetch_package_data(client, pkg_name)
            if data is not None:
                yield data


def nuget_source(package_names: typing.Union[str, Iterable[str]]):
    """DLT source to fetch package metadata from NuGet v3 API.

    Args:
        package_names: A package name or sequence of package names (e.g. ``"Newtonsoft.Json"``).
    """
    if isinstance(package_names, str):
        packages = [package_names]
    else:
        packages = list(package_names)

    if dlt is not None:
        @dlt.source
        def source():
            @dlt.resource(name="nuget_packages", write_disposition="replace")
            def get_packages():
                yield from _nuget_resource_generator(packages)

            return get_packages

        return source()

    return _nuget_resource_generator(packages)
