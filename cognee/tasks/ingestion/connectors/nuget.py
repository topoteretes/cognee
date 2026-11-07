"""NuGet v3 connector for cognee."""

from __future__ import annotations
from typing import Any, Iterable, Sequence
import httpx
import dlt

NUGET_V3_INDEX_URL = "https://api.nuget.org/v3/index.json"


class NuGetClient:
    def __init__(self, index_url: str = NUGET_V3_INDEX_URL, timeout: float = 30.0) -> None:
        self.index_url = index_url
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)
        self._registration_base: str | None = None

    def _get_registration_base(self) -> str:
        if self._registration_base:
            return self._registration_base
        resp = self._client.get(self.index_url)
        resp.raise_for_status()
        for res in resp.json().get("resources", []):
            rtype = res.get("@type", "")
            if isinstance(rtype, str) and rtype.startswith("RegistrationsBaseUrl"):
                self._registration_base = res["@id"].rstrip("/")
                return self._registration_base
        raise RuntimeError(f"No RegistrationsBaseUrl found in NuGet index {self.index_url}")

    def get_package_registration(self, package_id: str) -> dict[str, Any]:
        base = self._get_registration_base()
        url = f"{base}/{package_id.lower()}/index.json"
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.json()


def build_nuget_client(index_url: str = NUGET_V3_INDEX_URL) -> NuGetClient:
    return NuGetClient(index_url=index_url)


@dlt.source(name="nuget")
def nuget_source(
    package_ids: Sequence[str],
    index_url: str = NUGET_V3_INDEX_URL,
    client: NuGetClient | None = None,
):
    if not package_ids:
        raise ValueError("package_ids must contain at least one NuGet package ID")
    nuget = client or build_nuget_client(index_url=index_url)

    @dlt.resource(name="packages", primary_key="id", write_disposition="merge")
    def packages() -> Iterable[dict[str, Any]]:
        for pkg_id in package_ids:
            reg = nuget.get_package_registration(pkg_id)
            entries: list[dict[str, Any]] = []
            for page in reg.get("items", []):
                entries.extend(page.get("items", []))
            latest = entries[-1].get("catalogEntry", {}) if entries else {}
            version = latest.get("version", "")
            desc = latest.get("description") or latest.get("summary") or ""
            authors = latest.get("authors", "")
            yield {
                "id": f"nuget:{pkg_id.lower()}",
                "package_id": pkg_id,
                "latest_version": version,
                "authors": authors,
                "text": f"# {pkg_id} ({version})\nAuthors: {authors}\n\n{desc}".strip(),
            }

    return [packages]
