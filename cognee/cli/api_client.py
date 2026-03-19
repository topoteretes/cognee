"""Thin HTTP client that delegates CLI commands to a running Cognee API server.

When ``--api-url`` is supplied (e.g. ``--api-url http://localhost:8000``),
every CLI command is forwarded as an HTTP request to the server instead of
being executed in-process.  This avoids file-based database locking issues
(SQLite, KuzuDB, LanceDB) that arise when multiple CLI processes try to
access the same files concurrently.

The single API server process owns all database connections and serialises
writes, which is the correct model for file-based backends.
"""

from __future__ import annotations

import io
from typing import Any, Optional
from urllib.parse import urljoin


def _get_http_client():
    """Return an httpx client (imported lazily so the dep is optional)."""
    try:
        import httpx
    except ImportError:
        raise SystemExit(
            "The 'httpx' package is required for --api-url mode.  "
            "Install it with:  uv pip install httpx"
        )
    return httpx


class CogneeApiClient:
    """Stateless wrapper around the Cognee REST API."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- helpers ---------------------------------------------------------

    def _url(self, path: str) -> str:
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _raise_for_status(self, resp) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"API error {resp.status_code}: {detail}")

    # -- probes ----------------------------------------------------------

    def health(self) -> dict:
        httpx = _get_http_client()
        with httpx.Client(timeout=5.0) as c:
            r = c.get(self._url("/health"))
            self._raise_for_status(r)
            return r.json()

    # -- add -------------------------------------------------------------

    def add(
        self,
        data_items: list[str],
        dataset_name: str = "main_dataset",
    ) -> dict:
        httpx = _get_http_client()
        with httpx.Client(timeout=self.timeout) as c:
            files = []
            for item in data_items:
                # Send each text item as an in-memory "file upload"
                files.append(
                    ("data", (f"text_{len(files)}.txt", io.BytesIO(item.encode()), "text/plain"))
                )
            form_data = {"datasetName": dataset_name}
            r = c.post(self._url("/api/v1/add"), files=files, data=form_data)
            self._raise_for_status(r)
            return r.json()

    # -- cognify ---------------------------------------------------------

    def cognify(
        self,
        datasets: Optional[list[str]] = None,
        run_in_background: bool = False,
        chunks_per_batch: Optional[int] = None,
    ) -> dict:
        httpx = _get_http_client()
        payload: dict[str, Any] = {"run_in_background": run_in_background}
        if datasets:
            payload["datasets"] = datasets
        if chunks_per_batch is not None:
            payload["chunks_per_batch"] = chunks_per_batch
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(self._url("/api/v1/cognify"), json=payload)
            self._raise_for_status(r)
            return r.json()

    # -- search ----------------------------------------------------------

    def search(
        self,
        query: str,
        search_type: str = "GRAPH_COMPLETION",
        datasets: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list:
        httpx = _get_http_client()
        payload: dict[str, Any] = {
            "query": query,
            "search_type": search_type,
            "top_k": top_k,
        }
        if datasets:
            payload["datasets"] = datasets
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(self._url("/api/v1/search"), json=payload)
            self._raise_for_status(r)
            return r.json()

    # -- memify ----------------------------------------------------------

    def memify(
        self,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        data: Optional[str] = None,
        node_name: Optional[list[str]] = None,
        run_in_background: bool = False,
    ) -> dict:
        httpx = _get_http_client()
        payload: dict[str, Any] = {"run_in_background": run_in_background}
        if dataset_name:
            payload["dataset_name"] = dataset_name
        if dataset_id:
            payload["dataset_id"] = dataset_id
        if data:
            payload["data"] = data
        if node_name:
            payload["node_name"] = node_name
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(self._url("/api/v1/memify"), json=payload)
            self._raise_for_status(r)
            return r.json()

    # -- datasets --------------------------------------------------------

    def datasets_list(self) -> list[dict]:
        httpx = _get_http_client()
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(self._url("/api/v1/datasets"))
            self._raise_for_status(r)
            return r.json()

    def datasets_create(self, name: str) -> dict:
        httpx = _get_http_client()
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(self._url("/api/v1/datasets"), json={"name": name})
            self._raise_for_status(r)
            return r.json()

    def datasets_data(self, dataset_id: str) -> list[dict]:
        httpx = _get_http_client()
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(self._url(f"/api/v1/datasets/{dataset_id}/data"))
            self._raise_for_status(r)
            return r.json()

    def datasets_status(self, dataset_ids: list[str]) -> dict:
        httpx = _get_http_client()
        params = [("dataset", did) for did in dataset_ids]
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(self._url("/api/v1/datasets/status"), params=params)
            self._raise_for_status(r)
            return r.json()

    def datasets_graph(self, dataset_id: str) -> dict:
        httpx = _get_http_client()
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(self._url(f"/api/v1/datasets/{dataset_id}/graph"))
            self._raise_for_status(r)
            return r.json()

    def datasets_delete(self, dataset_id: str) -> None:
        httpx = _get_http_client()
        with httpx.Client(timeout=self.timeout) as c:
            r = c.delete(self._url(f"/api/v1/datasets/{dataset_id}"))
            self._raise_for_status(r)

    def datasets_delete_all(self) -> None:
        httpx = _get_http_client()
        with httpx.Client(timeout=self.timeout) as c:
            r = c.delete(self._url("/api/v1/datasets"))
            self._raise_for_status(r)
