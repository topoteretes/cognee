"""Thin HTTP client that delegates CLI commands to a running Cognee API server.

When ``--api-url`` is supplied (e.g. ``--api-url http://localhost:8000``),
every CLI command is forwarded as an HTTP request to the server instead of
being executed in-process.  This avoids file-based database locking issues
(SQLite, Ladybug, LanceDB) that arise when multiple CLI processes try to
access the same files concurrently.

The single API server process owns all database connections and serialises
writes, which is the correct model for file-based backends.
"""

from __future__ import annotations

import io
import mimetypes
import os
from typing import Any, Optional
from urllib.parse import urljoin


def _import_httpx():
    """Import httpx lazily so the dependency is optional."""
    try:
        import httpx

        return httpx
    except ImportError:
        raise SystemExit(
            "The 'httpx' package is required for --api-url mode.  "
            "Install it with:  uv pip install httpx"
        )


def is_connection_error(exc: BaseException) -> bool:
    """True if *exc* is a transport-level failure — i.e. we could not reach or
    get a response from the server (connect refused, DNS failure, timeout).

    This is deliberately narrower than "any error": an HTTP 4xx/5xx means the
    server *is* reachable, so those must not be reported as "cannot connect".
    """
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, httpx.TransportError)


class CogneeApiClient:
    """Wrapper around the Cognee REST API with a shared connection pool."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        headers: Optional[dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._extra_headers = headers or {}
        self._client = None

    # -- lifecycle -------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            httpx = _import_httpx()
            self._client = httpx.Client(
                timeout=self.timeout,
                headers=self._extra_headers,
                # Collection routes are canonicalised differently across
                # deployments (cloud serves "/api/v1/datasets/" with a slash,
                # local OSS serves it without), so a request to the other form
                # gets a 307.  Follow it automatically.  Safe now that the
                # server no longer downgrades the redirect to http (CLO-320).
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- helpers ---------------------------------------------------------

    def _url(self, path: str) -> str:
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _raise_for_status(self, resp) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            from cognee.cli.error_emit import parse_http_error_payload

            raise parse_http_error_payload(detail, resp.status_code)

    # -- probes ----------------------------------------------------------

    def health(self) -> dict:
        """Probe the server with a short timeout.

        Reuses the shared client so auth headers and redirect handling apply.
        Returns the parsed body for *any* HTTP response (including a 503 from a
        degraded backend) — a reachable server must never look like a transport
        failure.  Only genuine transport errors propagate to the caller.
        """
        r = self._get_client().get(self._url("/health"), timeout=5.0)
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}

    # -- add -------------------------------------------------------------

    def add(
        self,
        data_items: list[str],
        dataset_name: str = "main_dataset",
    ) -> dict:
        files = []
        opened = []
        try:
            for item in data_items:
                if os.path.isfile(item):
                    mime, _ = mimetypes.guess_type(item)
                    fh = open(item, "rb")  # noqa: SIM115
                    opened.append(fh)
                    files.append(
                        ("data", (os.path.basename(item), fh, mime or "application/octet-stream"))
                    )
                else:
                    files.append(
                        (
                            "data",
                            (f"text_{len(files)}.txt", io.BytesIO(item.encode()), "text/plain"),
                        )
                    )
            form_data = {"datasetName": dataset_name}
            r = self._get_client().post(self._url("/api/v1/add"), files=files, data=form_data)
            self._raise_for_status(r)
            return r.json()
        finally:
            for fh in opened:
                fh.close()

    # -- cognify ---------------------------------------------------------

    def cognify(
        self,
        datasets: Optional[list[str]] = None,
        run_in_background: bool = False,
        chunks_per_batch: Optional[int] = None,
    ) -> dict:
        payload: dict[str, Any] = {"run_in_background": run_in_background}
        if datasets:
            payload["datasets"] = datasets
        if chunks_per_batch is not None:
            payload["chunks_per_batch"] = chunks_per_batch
        r = self._get_client().post(self._url("/api/v1/cognify"), json=payload)
        self._raise_for_status(r)
        return r.json()

    # -- search ----------------------------------------------------------

    def search(
        self,
        query: str,
        search_type: str = "GRAPH_COMPLETION",
        datasets: Optional[list[str]] = None,
        top_k: int = 15,
    ) -> list:
        payload: dict[str, Any] = {
            "query": query,
            "search_type": search_type,
            "top_k": top_k,
        }
        if datasets:
            payload["datasets"] = datasets
        r = self._get_client().post(self._url("/api/v1/search"), json=payload)
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
        payload: dict[str, Any] = {"run_in_background": run_in_background}
        if dataset_name:
            payload["dataset_name"] = dataset_name
        if dataset_id:
            payload["dataset_id"] = dataset_id
        if data:
            payload["data"] = data
        if node_name:
            payload["node_name"] = node_name
        r = self._get_client().post(self._url("/api/v1/memify"), json=payload)
        self._raise_for_status(r)
        return r.json()

    # -- datasets --------------------------------------------------------

    def datasets_list(self) -> list[dict]:
        r = self._get_client().get(self._url("/api/v1/datasets/"))
        self._raise_for_status(r)
        return r.json()

    def datasets_create(self, name: str) -> dict:
        r = self._get_client().post(self._url("/api/v1/datasets/"), json={"name": name})
        self._raise_for_status(r)
        return r.json()

    def datasets_data(self, dataset_id: str) -> list[dict]:
        r = self._get_client().get(self._url(f"/api/v1/datasets/{dataset_id}/data"))
        self._raise_for_status(r)
        return r.json()

    def datasets_status(
        self, dataset_ids: list[str], pipelines: Optional[list[str]] = None
    ) -> dict:
        params = [("dataset", did) for did in dataset_ids]
        if pipelines:
            params.extend([("pipeline", pipeline) for pipeline in pipelines])
        r = self._get_client().get(self._url("/api/v1/datasets/status"), params=params)
        self._raise_for_status(r)
        return r.json()

    def datasets_graph(self, dataset_id: str) -> dict:
        r = self._get_client().get(self._url(f"/api/v1/datasets/{dataset_id}/graph"))
        self._raise_for_status(r)
        return r.json()

    def datasets_delete(self, dataset_id: str) -> None:
        r = self._get_client().delete(self._url(f"/api/v1/datasets/{dataset_id}"))
        self._raise_for_status(r)

    def datasets_delete_all(self) -> None:
        r = self._get_client().delete(self._url("/api/v1/datasets/"))
        self._raise_for_status(r)

    # -- remember --------------------------------------------------------

    def remember(
        self,
        data_items: list[str],
        dataset_name: str = "main_dataset",
        session_id: Optional[str] = None,
        node_set: Optional[list[str]] = None,
        run_in_background: bool = False,
        chunk_size: Optional[int] = None,
        chunks_per_batch: Optional[int] = None,
        custom_prompt: Optional[str] = None,
    ) -> dict:
        files = []
        opened = []
        try:
            for item in data_items:
                if os.path.isfile(item):
                    mime, _ = mimetypes.guess_type(item)
                    fh = open(item, "rb")  # noqa: SIM115
                    opened.append(fh)
                    files.append(
                        ("data", (os.path.basename(item), fh, mime or "application/octet-stream"))
                    )
                else:
                    files.append(
                        (
                            "data",
                            (f"text_{len(files)}.txt", io.BytesIO(item.encode()), "text/plain"),
                        )
                    )
            form_data: dict[str, Any] = {
                "datasetName": dataset_name,
                "run_in_background": "true" if run_in_background else "false",
            }
            if session_id:
                form_data["session_id"] = session_id
            if chunk_size is not None:
                form_data["chunk_size"] = str(chunk_size)
            if chunks_per_batch is not None:
                form_data["chunks_per_batch"] = str(chunks_per_batch)
            if custom_prompt:
                form_data["custom_prompt"] = custom_prompt
            if node_set:
                form_data["node_set"] = node_set
            r = self._get_client().post(
                self._url("/api/v1/remember"),
                files=files,
                data=form_data,
            )
            self._raise_for_status(r)
            return r.json()
        finally:
            for fh in opened:
                fh.close()

    # -- recall ----------------------------------------------------------

    def recall(
        self,
        query: str,
        search_type: Optional[str] = "GRAPH_COMPLETION",
        datasets: Optional[list[str]] = None,
        top_k: int = 15,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        node_name: Optional[list[str]] = None,
        only_context: bool = False,
        verbose: bool = False,
    ) -> list:
        # search_type=None opts the server into auto-routing (session-only
        # mode when session_id is set, graph otherwise).
        payload: dict[str, Any] = {
            "query": query,
            "search_type": search_type,
            "top_k": top_k,
            "only_context": only_context,
            "verbose": verbose,
        }
        if datasets:
            payload["datasets"] = datasets
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if session_id:
            payload["session_id"] = session_id
        if node_name:
            payload["node_name"] = node_name
        r = self._get_client().post(self._url("/api/v1/recall"), json=payload)
        self._raise_for_status(r)
        return r.json()

    # -- improve ---------------------------------------------------------

    def improve(
        self,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        node_name: Optional[list[str]] = None,
        session_ids: Optional[list[str]] = None,
        run_in_background: bool = False,
    ) -> dict:
        payload: dict[str, Any] = {"run_in_background": run_in_background}
        if dataset_name:
            payload["dataset_name"] = dataset_name
        if dataset_id:
            payload["dataset_id"] = str(dataset_id)
        if node_name:
            payload["node_name"] = node_name
        if session_ids:
            payload["session_ids"] = session_ids
        r = self._get_client().post(self._url("/api/v1/improve"), json=payload)
        self._raise_for_status(r)
        return r.json()

    # -- forget ----------------------------------------------------------

    def forget(
        self,
        dataset: Optional[str] = None,
        dataset_id: Optional[str] = None,
        data_id: Optional[str] = None,
        everything: bool = False,
        memory_only: bool = False,
    ) -> dict:
        payload: dict[str, Any] = {
            "everything": everything,
            "memory_only": memory_only,
        }
        if dataset is not None:
            payload["dataset"] = str(dataset)
        if dataset_id is not None:
            payload["dataset_id"] = str(dataset_id)
        if data_id is not None:
            payload["data_id"] = str(data_id)
        r = self._get_client().post(self._url("/api/v1/forget"), json=payload)
        self._raise_for_status(r)
        return r.json()
