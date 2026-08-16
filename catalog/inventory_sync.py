"""Cross-repo drift check between the catalog and cognee-integrations/inventory.yml.

Fetches ``inventory.yml`` from ``topoteretes/cognee-integrations`` via the
GitHub REST API and cross-checks the ``slug`` values against every
``inventory_slug`` field in the local catalog.

Two directions of drift are reported:

* **Coverage gaps**: slugs present in ``inventory.yml`` that no catalog entry
  claims. These are integrations the community has surfaced without a catalog
  card, so users won't find them in the Hub.
* **Stale references**: ``inventory_slug`` values in catalog entries that no
  longer exist in ``inventory.yml``. These usually mean an integration was
  renamed or removed upstream and the catalog didn't catch up.

Kept separate from :mod:`catalog.loader` so ``python -m catalog.loader``
stays offline. This module is invoked by CI (which has network) and can be
run locally by anyone wanting to check drift before opening a PR.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from typing import Any

import yaml

from catalog.loader import CatalogError, load_catalog

INVENTORY_REPO = "topoteretes/cognee-integrations"
INVENTORY_PATH = "integrations/inventory.yml"
GITHUB_API = "https://api.github.com"
USER_AGENT = "cognee-catalog-drift-check"


class InventoryFetchError(RuntimeError):
    """Raised when the upstream ``inventory.yml`` cannot be fetched."""


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_inventory() -> dict[str, Any]:
    """Fetch and parse the upstream ``inventory.yml``.

    Uses ``GITHUB_TOKEN`` when set (avoids anonymous rate limits in CI). Any
    network, decode, or parse failure is re-raised as :class:`InventoryFetchError`
    so callers get one clean error type instead of a raw traceback.
    """

    url = f"{GITHUB_API}/repos/{INVENTORY_REPO}/contents/{INVENTORY_PATH}"
    request = urllib.request.Request(url, headers=_github_headers())
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
        content = payload.get("content")
        encoding = payload.get("encoding")
        if not content or encoding != "base64":
            raise InventoryFetchError("inventory.yml payload was not base64-encoded content")
        parsed = yaml.safe_load(base64.b64decode(content).decode("utf-8"))
    except InventoryFetchError:
        raise
    except (OSError, ValueError, yaml.YAMLError) as cause:
        raise InventoryFetchError(f"could not fetch or parse inventory.yml: {cause}") from cause

    if not isinstance(parsed, dict):
        raise InventoryFetchError("inventory.yml top-level was not a mapping")
    return parsed


def collect_inventory_slugs(inventory: dict[str, Any]) -> set[str]:
    entries = inventory.get("integrations")
    if not isinstance(entries, list):
        raise InventoryFetchError("inventory.yml: expected `integrations` list")

    slugs: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug")
        if isinstance(slug, str) and slug:
            slugs.add(slug)
    return slugs


def collect_catalog_inventory_slugs() -> dict[str, str]:
    """Map ``inventory_slug`` values to their catalog entry ids."""

    catalog = load_catalog()
    mapped: dict[str, str] = {}
    for entry in catalog:
        if entry.inventory_slug is not None:
            mapped[entry.inventory_slug] = entry.id
    return mapped


def report_drift(inventory_slugs: set[str], catalog_slugs: dict[str, str]) -> list[str]:
    """Return a list of drift descriptions. Empty list means no drift."""

    problems: list[str] = []

    uncovered = sorted(inventory_slugs - set(catalog_slugs))
    for slug in uncovered:
        problems.append(
            f"coverage gap: inventory.yml slug '{slug}' has no catalog entry "
            f"(add catalog/entries/integrations/{slug}.yaml with inventory_slug: {slug})"
        )

    stale = sorted(set(catalog_slugs) - inventory_slugs)
    for slug in stale:
        problems.append(
            f"stale reference: catalog entry '{catalog_slugs[slug]}' claims "
            f"inventory_slug '{slug}', which is not in inventory.yml"
        )

    return problems


def main() -> int:
    """Report catalog/inventory drift.

    Exit codes: 0 when in sync or only coverage gaps remain; 1 on stale
    references (a catalog entry claims a slug the inventory no longer has);
    2 when the inventory could not be fetched. The CI step runs non-blocking,
    so an upstream reshape or a transient fetch failure never fails a PR.
    """

    try:
        inventory = fetch_inventory()
        inventory_slugs = collect_inventory_slugs(inventory)
    except InventoryFetchError as cause:
        print(f"error: {cause}", file=sys.stderr)
        return 2

    try:
        catalog_slugs = collect_catalog_inventory_slugs()
    except CatalogError as cause:
        print(str(cause), file=sys.stderr)
        return 1

    problems = report_drift(inventory_slugs, catalog_slugs)

    if not problems:
        print(
            f"catalog is in sync with {INVENTORY_REPO}/{INVENTORY_PATH} "
            f"({len(inventory_slugs)} inventory slugs, {len(catalog_slugs)} catalog references)"
        )
        return 0

    print(f"drift detected against {INVENTORY_REPO}/{INVENTORY_PATH}:")
    stale_seen = False
    for problem in problems:
        print(f"  - {problem}")
        if problem.startswith("stale reference:"):
            stale_seen = True

    if stale_seen:
        return 1

    print("(coverage gaps only; informational, not a failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
