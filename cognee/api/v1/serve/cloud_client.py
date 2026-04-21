"""Remote HTTP client that proxies V2 operations to a Cognee Cloud instance."""

import io
from typing import Any, Optional
from uuid import UUID

import aiohttp

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.cloud_client")


class CloudClient:
    """Async HTTP client for a remote Cognee Cloud tenant instance.

    All requests use ``X-Api-Key`` for authentication, matching the
    SaaS backend's API key auth backend.
    """

    def __init__(self, service_url: str, api_key: str):
        self.service_url = service_url.rstrip("/")
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-Api-Key": self.api_key},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _health_check(self) -> bool:
        """Verify the remote instance is reachable."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.service_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    # ----- V2 Operations -----

    async def remember(self, data: Any, dataset_name: str = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/remember — ingest data and build knowledge graph."""
        session = await self._get_session()

        form = aiohttp.FormData()
        form.add_field("datasetName", dataset_name)

        if kwargs.get("session_id"):
            form.add_field("session_id", kwargs["session_id"])
        if kwargs.get("run_in_background"):
            form.add_field("run_in_background", "true")
        if kwargs.get("custom_prompt"):
            form.add_field("custom_prompt", kwargs["custom_prompt"])

        # Handle data — string or file-like objects
        if isinstance(data, str):
            form.add_field(
                "data",
                io.BytesIO(data.encode("utf-8")),
                filename="data.txt",
                content_type="text/plain",
            )
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    form.add_field(
                        "data",
                        io.BytesIO(item.encode("utf-8")),
                        filename="data.txt",
                        content_type="text/plain",
                    )
                elif hasattr(item, "read"):
                    name = getattr(item, "name", "upload")
                    form.add_field("data", item, filename=name)
        elif hasattr(data, "read"):
            name = getattr(data, "name", "upload")
            form.add_field("data", data, filename=name)

        async with session.post(f"{self.service_url}/api/v1/remember", data=form) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote remember failed ({resp.status}): {body}")
            return await resp.json()

    async def remember_entry(
        self,
        entry,
        dataset_name: str = "main_dataset",
        session_id: Optional[str] = None,
    ) -> dict:
        """POST /api/v1/remember/entry — store a typed MemoryEntry in session cache.

        ``entry`` is a pydantic MemoryEntry (QAEntry / TraceEntry / FeedbackEntry).
        """
        if session_id is None:
            raise ValueError("session_id is required for typed memory entries")

        session = await self._get_session()

        # Pydantic v2: model_dump preserves the discriminator field.
        entry_dump = entry.model_dump(mode="json")

        payload = {
            "entry": entry_dump,
            "dataset_name": dataset_name,
            "session_id": session_id,
        }

        async with session.post(
            f"{self.service_url}/api/v1/remember/entry",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote remember_entry failed ({resp.status}): {body}")
            return await resp.json()

    async def recall(self, query_text: str, query_type: Optional[str] = None, **kwargs) -> list:
        """POST /api/v1/recall — query the knowledge graph and/or session cache."""
        session = await self._get_session()

        payload: dict = {"query": query_text}
        if query_type:
            payload["search_type"] = query_type if isinstance(query_type, str) else query_type.value
        if kwargs.get("datasets"):
            payload["datasets"] = kwargs["datasets"]
        if kwargs.get("top_k"):
            payload["top_k"] = kwargs["top_k"]
        if kwargs.get("system_prompt"):
            payload["system_prompt"] = kwargs["system_prompt"]
        if kwargs.get("session_id"):
            payload["session_id"] = kwargs["session_id"]
        if kwargs.get("scope") is not None:
            payload["scope"] = kwargs["scope"]

        async with session.post(
            f"{self.service_url}/api/v1/recall",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote recall failed ({resp.status}): {body}")
            return await resp.json()

    async def improve(self, dataset: Any = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/improve — enrich the knowledge graph."""
        session = await self._get_session()

        payload = {}
        if isinstance(dataset, UUID):
            payload["dataset_id"] = str(dataset)
        else:
            payload["dataset_name"] = str(dataset)
        if kwargs.get("run_in_background"):
            payload["run_in_background"] = True
        if kwargs.get("node_name"):
            payload["node_name"] = kwargs["node_name"]

        async with session.post(
            f"{self.service_url}/api/v1/improve",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote improve failed ({resp.status}): {body}")
            return await resp.json()

    # ----- V1 Operations (add / cognify / search) -----

    async def add(self, data: Any, dataset_name: str = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/add — ingest data into a dataset."""
        session = await self._get_session()

        form = aiohttp.FormData()
        form.add_field("datasetName", dataset_name)

        if isinstance(data, str):
            form.add_field(
                "data",
                io.BytesIO(data.encode("utf-8")),
                filename="data.txt",
                content_type="text/plain",
            )
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    form.add_field(
                        "data",
                        io.BytesIO(item.encode("utf-8")),
                        filename="data.txt",
                        content_type="text/plain",
                    )
                elif hasattr(item, "read"):
                    name = getattr(item, "name", "upload")
                    form.add_field("data", item, filename=name)
        elif hasattr(data, "read"):
            name = getattr(data, "name", "upload")
            form.add_field("data", data, filename=name)

        async with session.post(f"{self.service_url}/api/v1/add", data=form) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote add failed ({resp.status}): {body}")
            return await resp.json()

    async def cognify(self, datasets: Any = None, **kwargs) -> dict:
        """POST /api/v1/cognify — build the knowledge graph."""
        session = await self._get_session()

        payload: dict = {}
        if datasets:
            payload["datasets"] = (
                [str(d) for d in datasets] if isinstance(datasets, list) else [str(datasets)]
            )

        async with session.post(
            f"{self.service_url}/api/v1/cognify",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote cognify failed ({resp.status}): {body}")
            return await resp.json()

    async def search(self, query: str, **kwargs) -> list:
        """POST /api/v1/search — query the knowledge graph."""
        session = await self._get_session()

        payload: dict = {"query": query}
        if kwargs.get("search_type"):
            st = kwargs["search_type"]
            payload["searchType"] = st if isinstance(st, str) else st.value
        if kwargs.get("datasets"):
            payload["datasets"] = kwargs["datasets"]

        async with session.post(
            f"{self.service_url}/api/v1/search",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote search failed ({resp.status}): {body}")
            return await resp.json()

    async def forget(self, **kwargs) -> dict:
        """POST /api/v1/forget — delete data from the knowledge graph."""
        session = await self._get_session()

        payload = {}
        if kwargs.get("everything"):
            payload["everything"] = True
        if kwargs.get("dataset"):
            ds = kwargs["dataset"]
            payload["dataset"] = str(ds)
        if kwargs.get("data_id"):
            payload["data_id"] = str(kwargs["data_id"])

        async with session.post(
            f"{self.service_url}/api/v1/forget",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote forget failed ({resp.status}): {body}")
            return await resp.json()
