"""Ingest Mintlify MDX docs from a GitHub repo into Cognee's org dataset.

We read the raw MDX source (NOT the Mintlify chatbot API) so Cognee can
structure the docs as a graph without spending Mintlify credits.

Usage:
    python -m community_bot.ingest.docs
    # or from this folder:
    cd community-bot && python -m ingest.docs
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import httpx

# Ensure we can import config when run as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cognee  # noqa: E402

from config import (  # noqa: E402
    DOCS_REPO_BRANCH,
    DOCS_REPO_NAME,
    DOCS_REPO_OWNER,
    GITHUB_TOKEN,
    MAX_MDX_FILES,
    ORG_DATASET,
)

GITHUB_API = "https://api.github.com"


def _gh_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


async def list_mdx_paths(client: httpx.AsyncClient) -> list[dict]:
    """Return tree entries for every .mdx file in the docs repo."""
    url = (
        f"{GITHUB_API}/repos/{DOCS_REPO_OWNER}/{DOCS_REPO_NAME}"
        f"/git/trees/{DOCS_REPO_BRANCH}?recursive=1"
    )
    resp = await client.get(url, headers=_gh_headers())
    resp.raise_for_status()
    tree = resp.json().get("tree", [])
    mdx_entries = [
        e for e in tree if e.get("type") == "blob" and e.get("path", "").endswith(".mdx")
    ]
    return mdx_entries


async def fetch_mdx_content(client: httpx.AsyncClient, path: str) -> str:
    """Fetch the raw text content of a single file in the docs repo."""
    url = (
        f"{GITHUB_API}/repos/{DOCS_REPO_OWNER}/{DOCS_REPO_NAME}"
        f"/contents/{path}?ref={DOCS_REPO_BRANCH}"
    )
    resp = await client.get(url, headers=_gh_headers())
    resp.raise_for_status()
    data = resp.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    # Fallback: fetch raw via download_url
    raw = await client.get(data["download_url"], headers=_gh_headers())
    raw.raise_for_status()
    return raw.text


def _format_doc(path: str, content: str) -> str:
    return f"# Doc: {path}\nSource: {DOCS_REPO_OWNER}/{DOCS_REPO_NAME}\n\n{content}"


async def ingest_docs() -> int:
    async with httpx.AsyncClient(timeout=60.0) as client:
        entries = await list_mdx_paths(client)
        print(f"[docs] Found {len(entries)} .mdx files in {DOCS_REPO_OWNER}/{DOCS_REPO_NAME}")
        if MAX_MDX_FILES > 0:
            entries = entries[:MAX_MDX_FILES]
            print(f"[docs] Limiting to first {len(entries)} files (MAX_MDX_FILES)")

        ingested = 0
        for entry in entries:
            path = entry["path"]
            try:
                content = await fetch_mdx_content(client, path)
            except httpx.HTTPError as exc:
                print(f"[docs] ! failed to fetch {path}: {exc}")
                continue

            text = _format_doc(path, content)
            await cognee.add(text, dataset_name=ORG_DATASET)
            ingested += 1
            print(f"[docs] added ({ingested}/{len(entries)}) {path}")

    print(f"[docs] Running cognify on '{ORG_DATASET}' ...")
    await cognee.cognify(datasets=[ORG_DATASET])
    print(f"[docs] Done. Ingested {ingested} MDX files.")
    return ingested


if __name__ == "__main__":
    asyncio.run(ingest_docs())
