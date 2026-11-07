"""Packagist / Composer connector for cognee."""

from __future__ import annotations
from typing import Any, Iterable, Sequence
import httpx
import dlt

PACKAGIST_API_URL = "https://packagist.org"


class PackagistService:
    def __init__(self, base_url: str = PACKAGIST_API_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def get_package(self, package_name: str) -> dict[str, Any]:
        if "/" not in package_name:
            raise ValueError(f"Invalid Packagist package name '{package_name}', expected 'vendor/package'")
        resp = self._client.get(f"{self.base_url}/packages/{package_name}.json")
        resp.raise_for_status()
        return resp.json().get("package") or {}

    def get_advisories(self, package_name: str) -> list[dict[str, Any]]:
        resp = self._client.get(
            f"{self.base_url}/api/security-advisories/",
            params={"packages[]": package_name},
        )
        resp.raise_for_status()
        advisories = resp.json().get("advisories") or {}
        return advisories.get(package_name) or []


def build_packagist_service(base_url: str = PACKAGIST_API_URL) -> PackagistService:
    return PackagistService(base_url=base_url)


@dlt.source(name="packagist")
def packagist_source(
    packages: Sequence[str],
    include_advisories: bool = True,
    service: PackagistService | None = None,
):
    if not packages:
        raise ValueError("packages must contain at least one 'vendor/package' string")
    svc = service or build_packagist_service()

    @dlt.resource(name="package_documents", primary_key="id", write_disposition="merge")
    def package_documents() -> Iterable[dict[str, Any]]:
        for pkg_name in packages:
            pkg = svc.get_package(pkg_name)
            desc = pkg.get("description") or ""
            repo = pkg.get("repository") or ""
            versions = list((pkg.get("versions") or {}).keys())[:20]
            text_lines = [
                f"# {pkg.get('name', pkg_name)}",
                f"Repository: {repo}",
                f"Description: {desc}",
                f"Versions: {', '.join(versions)}",
            ]
            if include_advisories:
                for adv in svc.get_advisories(pkg_name):
                    text_lines.append(f"- Advisory {adv.get('cve') or adv.get('advisoryId')}: {adv.get('title')}")
            yield {
                "id": f"packagist:{pkg_name.lower()}",
                "package": pkg_name,
                "description": desc,
                "text": "\n".join(text_lines).strip(),
            }

    return [package_documents]
