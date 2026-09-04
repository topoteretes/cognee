"""Remote HTTP client that proxies V2 operations to a Cognee Cloud instance."""

import io
import json
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import aiohttp

from cognee.modules.search.types import ContextFormat
from cognee.modules.ingestion.data_types.TextData import create_text_data
from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.cloud_client")


def _text_upload_filename(text: str) -> str:
    """Content-hash filename for raw-text uploads, via local ingestion's namer.

    Delegates to ``TextData`` — the same source ``save_data_to_file`` uses for
    nameless text (``text_<md5>.txt``). A fixed placeholder here instead makes
    every text upload for a tenant collide on one remote object, racing
    concurrent adds against the server's content-hash read-back
    (FileContentHashingError 409s).
    """
    return create_text_data(text).get_metadata()["name"]


def _is_data_item(value: Any) -> bool:
    """A ``DataItem`` wrapper (duck-typed to avoid importing the tasks package here)."""
    return (
        hasattr(value, "data")
        and hasattr(value, "data_id")
        and not hasattr(value, "read")
        and not isinstance(value, (str, Path))
    )


def _local_file_path(value: Any) -> Optional[Path]:
    """The local file a ``/abs/path`` / ``file://`` / ``Path`` value names, else None."""
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, str) and (value.startswith("file://") or value.startswith("/")):
        raw = value[len("file://") :] if value.startswith("file://") else value
        return Path(raw).expanduser()
    return None


def _upload_field(form: "aiohttp.FormData", item: Any, *, strict_paths: bool) -> None:
    """Add one payload to ``form`` as a multipart ``data`` field.

    File-like objects stream as-is. A local path (``/abs/path``,
    ``file://...``, ``Path``) is read and its bytes uploaded — the server
    cannot see the caller's filesystem. With ``strict_paths`` a path that
    names no file raises; otherwise the string falls back to raw text, which
    is what add()/remember() always did with strings. Any other string is raw
    text, named by content hash like local ingestion names it.
    """
    if hasattr(item, "read"):
        name = Path(getattr(item, "name", "upload")).name or "upload"
        form.add_field("data", item, filename=name)
        return

    path = _local_file_path(item)
    if path is not None:
        if path.is_file():
            form.add_field("data", path.open("rb"), filename=path.name)
            return
        if strict_paths:
            raise FileNotFoundError(f"Upload source not found: {item}")

    text = str(item)
    form.add_field(
        "data",
        io.BytesIO(text.encode("utf-8")),
        filename=_text_upload_filename(text),
        content_type="text/plain",
    )


def _attach_upload(form: "aiohttp.FormData", data: Any) -> None:
    """Add exactly one document to ``form`` for the update route.

    Accepts what the local ``update()`` accepts: a ``DataItem`` (unwrapped to
    its payload — its id is pinned by the route's query param and it has no
    slot for label/external_metadata), a file-like object, a local path, or
    raw text. A single-item list is unwrapped; anything longer is a caller
    error, as it is locally.
    """
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"update() replaces exactly one document; got {len(data)} items.")
        data = data[0]
    if _is_data_item(data):
        data = data.data
    _upload_field(form, data, strict_paths=True)


def _attach_uploads(form: "aiohttp.FormData", data: Any) -> list:
    """Add every payload in ``data`` to ``form`` and carry DataItem attributes along.

    ``DataItem`` wrappers are unwrapped for the upload; their ``label``,
    ``external_metadata`` and ``data_id`` are sent as the positional JSON
    array fields the add/remember routes accept (one entry per file, null to
    skip), so a pinned id survives the wire and the server stores the document
    under it. Returns the pinned ids in upload order (None where unpinned).
    """
    items = data if isinstance(data, list) else [data]
    labels, metadata, data_ids = [], [], []
    for item in items:
        if _is_data_item(item):
            labels.append(item.label)
            metadata.append(item.external_metadata)
            data_ids.append(str(item.data_id) if item.data_id is not None else None)
            item = item.data
        else:
            labels.append(None)
            metadata.append(None)
            data_ids.append(None)
        _upload_field(form, item, strict_paths=False)

    if any(labels):
        form.add_field("labels", json.dumps(labels))
    if any(metadata):
        form.add_field("external_metadata", json.dumps(metadata))
    if any(data_ids):
        form.add_field("data_ids", json.dumps(data_ids))
    return data_ids


def _returned_data_ids(response: Any) -> set:
    """Data ids an add/remember response reports, as strings.

    remember() responses list them as ``items[].id``; add() responses (a
    pipeline run) as ``data_ingestion_info[].data_id``. Empty when the shape
    carries none (background runs, older servers).
    """
    found = set()
    if not isinstance(response, dict):
        return found
    for entry in response.get("items") or []:
        if isinstance(entry, dict) and entry.get("id") is not None:
            found.add(str(entry["id"]))
    for entry in response.get("data_ingestion_info") or []:
        if isinstance(entry, dict) and entry.get("data_id") is not None:
            found.add(str(entry["data_id"]))
    return found


def _verify_pinned_ids(response: Any, pinned: list, operation: str) -> None:
    """Fail loudly if the server minted its own ids instead of honoring the pins.

    An older server without the ``data_ids`` field ignores it silently
    (FastAPI drops unknown form fields), and the caller would go on to
    ``update()`` an id that does not exist remotely. When the response
    reports ids and a pin is missing from them, that is what happened.
    A response that reports no ids at all cannot be checked; it is logged.
    """
    requested = [pin for pin in pinned if pin]
    if not requested:
        return
    returned = _returned_data_ids(response)
    if not returned:
        logger.warning(
            "Remote %s response reports no data ids; cannot confirm the pinned data_id(s) %s "
            "were honored (background run, or a server without data_ids support)",
            operation,
            ", ".join(requested),
        )
        return
    missing = [pin for pin in requested if pin not in returned]
    if missing:
        raise RuntimeError(
            f"Remote {operation} did not honor pinned data_id(s) {', '.join(missing)}: the "
            f"server stored the document(s) under {', '.join(sorted(returned))}. The remote "
            "instance is probably older than this SDK and lacks the data_ids field on "
            f"POST /api/v1/{operation}; upgrade it or use the server-assigned id."
        )


def _node_set_tags(node_set: Any) -> list:
    """node_set as repeated-form-field values: a str is one tag, a list many."""
    if not node_set:
        return []
    if isinstance(node_set, str):
        return [node_set]
    return [str(tag) for tag in node_set if tag]


class CloudClient:
    """Async HTTP client for a remote Cognee Cloud tenant instance.

    All requests use ``X-Api-Key`` for authentication, matching the
    SaaS backend's API key auth backend.
    """

    def __init__(self, service_url: str, api_key: str):
        self.service_url = service_url.rstrip("/")
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    # Default for ordinary API calls: aiohttp's standard 5-minute total,
    # with connect failures surfacing quickly.
    # 600s: long-running blocking operations (e.g. cognify over a large
    # dataset) can legitimately take many minutes server-side
    DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=600, sock_connect=30)
    # Archive uploads (cognee.push) plus the synchronous server-side import
    # can legitimately exceed any fixed total; per-read inactivity stays
    # bounded instead. Applied per-request, only to archive uploads.
    UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-Api-Key": self.api_key},
                timeout=self.DEFAULT_TIMEOUT,
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

    async def _auth_check(self) -> Optional[int]:
        """Status of an authenticated probe, or None when unreachable.

        ``/health`` is unauthenticated, so it cannot tell a working API key
        from a rejected one. Probing an authenticated endpoint lets serve()
        fail at connect time instead of on the first real operation.
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.service_url}/api/v1/datasets") as resp:
                return resp.status
        except Exception:
            return None

    # ----- V2 Operations -----

    async def remember(self, data: Any, dataset_name: str = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/remember — ingest data and build knowledge graph."""
        session = await self._get_session()

        form = aiohttp.FormData()
        form.add_field("datasetName", dataset_name)

        if kwargs.get("dataset_id"):
            form.add_field("datasetId", str(kwargs["dataset_id"]))
        if kwargs.get("session_id"):
            form.add_field("session_id", kwargs["session_id"])
        if kwargs.get("run_in_background"):
            form.add_field("run_in_background", "true")
        if kwargs.get("custom_prompt"):
            form.add_field("custom_prompt", kwargs["custom_prompt"])
        if kwargs.get("chunk_size") is not None:
            form.add_field("chunk_size", str(kwargs["chunk_size"]))
        if kwargs.get("chunks_per_batch") is not None:
            form.add_field("chunks_per_batch", str(kwargs["chunks_per_batch"]))
        content_type_kw = kwargs.get("content_type")
        if content_type_kw is not None:
            form.add_field("content_type", str(content_type_kw))
        if kwargs.get("import_mode") is not None:
            form.add_field("import_mode", str(kwargs["import_mode"]))
        for tag in _node_set_tags(kwargs.get("node_set")):
            form.add_field("node_set", tag)

        pinned_ids: list = []
        # Code repos travel as spec strings in the 'repositories' form field —
        # the server clones git URLs itself and reads local paths from its own
        # filesystem (only useful when it shares the caller's filesystem).
        # Nothing is uploaded.
        if content_type_kw == "code":
            specs = data if isinstance(data, list) else [data]
            for spec in specs:
                form.add_field("repositories", str(spec))
            if kwargs.get("index_vectors"):
                form.add_field("index_vectors", "true")
        # Skills are local SKILL.md files. The server's add_skills() reads
        # paths from its own filesystem — sending the path string verbatim
        # would have the server look for that path on the POD, not the
        # caller. For content_type="skills", read each SKILL.md and upload
        # its bytes so the server can write them to a tempdir.
        elif content_type_kw == "skills" and isinstance(data, (str, Path)):
            source = Path(data).expanduser()
            if source.is_file():
                skill_files = [source] if source.name == "SKILL.md" else []
            elif source.is_dir():
                skill_files = sorted(source.rglob("SKILL.md"))
            else:
                raise FileNotFoundError(f"Skills source not found: {data}")
            if not skill_files:
                raise ValueError(f"No SKILL.md files under {data}")
            base = source if source.is_dir() else source.parent
            for skill_path in skill_files:
                # Preserve relative structure so the server can reconstruct
                # the SKILL.md layout when writing to its tempdir.
                rel = skill_path.relative_to(base).as_posix()
                form.add_field("data", skill_path.open("rb"), filename=rel)
        # Normal ingestion — strings, local paths, file-like objects, or
        # DataItems carrying label / external_metadata / a pinned data_id.
        else:
            pinned_ids = _attach_uploads(form, data)

        # Code ingestion can block on a clone + whole-repo parse; the archive
        # timeout (no total cap) fits both. Prefer run_in_background=True for
        # large repos regardless.
        timeout = (
            self.UPLOAD_TIMEOUT
            if kwargs.get("content_type") in ("cogx-archive", "code")
            else self.DEFAULT_TIMEOUT
        )
        async with session.post(
            f"{self.service_url}/api/v1/remember", data=form, timeout=timeout
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote remember failed ({resp.status}): {body}")
            result = await resp.json()
        _verify_pinned_ids(result, pinned_ids, "remember")
        return result

    async def remember_entry(
        self,
        entry,
        dataset_name: str = "main_dataset",
        session_id: Optional[str] = None,
        skill_improvement: Optional[dict] = None,
    ) -> dict:
        """POST /api/v1/remember/entry — store a typed MemoryEntry.

        ``entry`` is a pydantic MemoryEntry.
        """
        session = await self._get_session()

        # Pydantic v2: model_dump preserves the discriminator field.
        entry_dump = entry.model_dump(mode="json")

        payload = {
            "entry": entry_dump,
            "dataset_name": dataset_name,
            "session_id": session_id,
            "skill_improvement": skill_improvement,
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
        if kwargs.get("dataset_ids"):
            payload["dataset_ids"] = [str(dataset_id) for dataset_id in kwargs["dataset_ids"]]
        elif kwargs.get("datasets"):
            payload["datasets"] = kwargs["datasets"]
        if kwargs.get("top_k"):
            payload["top_k"] = kwargs["top_k"]
        if kwargs.get("system_prompt"):
            payload["system_prompt"] = kwargs["system_prompt"]
        if kwargs.get("node_name"):
            payload["node_name"] = kwargs["node_name"]
        if kwargs.get("only_context"):
            payload["only_context"] = kwargs["only_context"]
        # Only the non-default shape is worth sending: an older instance ignores the
        # field, and omitting it keeps the request identical to what it always was.
        if ContextFormat.parse(kwargs.get("context_format")) is ContextFormat.PROMPT:
            payload["context_format"] = ContextFormat.PROMPT.value
        if kwargs.get("verbose"):
            payload["verbose"] = kwargs["verbose"]
        if kwargs.get("session_id"):
            payload["session_id"] = kwargs["session_id"]
        if kwargs.get("scope") is not None:
            payload["scope"] = kwargs["scope"]
        if kwargs.get("context_profile") is not None:
            payload["context_profile"] = kwargs["context_profile"]
        if kwargs.get("include_references") is not None:
            payload["include_references"] = kwargs["include_references"]
        if kwargs.get("response_schema") is not None:
            payload["response_schema"] = kwargs["response_schema"]
        if kwargs.get("tool_connections") is not None:
            payload["tool_connections"] = kwargs["tool_connections"]
        if kwargs.get("tools_trigger") not in (None, "always"):
            payload["tools_trigger"] = kwargs["tools_trigger"]
        if kwargs.get("code_query") is not None:
            payload["code_query"] = kwargs["code_query"]

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
        if dataset_name:
            form.add_field("datasetName", dataset_name)
        if kwargs.get("dataset_id"):
            form.add_field("datasetId", str(kwargs["dataset_id"]))
        for tag in _node_set_tags(kwargs.get("node_set")):
            form.add_field("node_set", tag)
        if kwargs.get("run_in_background"):
            form.add_field("run_in_background", "true")

        pinned_ids = _attach_uploads(form, data)

        async with session.post(f"{self.service_url}/api/v1/add", data=form) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote add failed ({resp.status}): {body}")
            result = await resp.json()
        _verify_pinned_ids(result, pinned_ids, "add")
        return result

    async def update(
        self,
        data_id: UUID,
        data: Any,
        dataset_id: Optional[UUID] = None,
        node_set: Optional[list] = None,
        chunk_level_diff: bool = True,
        dataset_name: Optional[str] = None,
    ) -> dict:
        """PATCH /api/v1/update — replace one document in place on the remote.

        Mirrors the server route: ``data_id``, the dataset (``dataset_id`` or
        ``dataset_name``, exactly one) and ``chunk_level_diff`` travel as query
        params, the new content as the multipart ``data`` file, ``node_set`` as
        repeated form fields. The server keeps the document's id across the
        update, so this is a real replace — never a local delete plus a remote
        add, which would mint a fresh id on the remote and leave the original
        in place.
        """
        if (dataset_id is None) == (dataset_name is None):
            raise ValueError("update() takes exactly one of dataset_id or dataset_name.")

        session = await self._get_session()

        form = aiohttp.FormData()
        _attach_upload(form, data)
        for tag in _node_set_tags(node_set):
            form.add_field("node_set", tag)

        params = {
            "data_id": str(data_id),
            "chunk_level_diff": "true" if chunk_level_diff else "false",
        }
        if dataset_id is not None:
            params["dataset_id"] = str(dataset_id)
        else:
            params["dataset_name"] = dataset_name

        async with session.patch(
            f"{self.service_url}/api/v1/update", params=params, data=form
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote update failed ({resp.status}): {body}")
            return await resp.json()

    async def list_data(self, dataset_id: UUID) -> list:
        """GET /api/v1/datasets/{dataset_id}/data — list the documents in a dataset."""
        session = await self._get_session()

        async with session.get(f"{self.service_url}/api/v1/datasets/{dataset_id}/data") as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote list_data failed ({resp.status}): {body}")
            return await resp.json()

    async def cognify(self, datasets: Any = None, **kwargs) -> dict:
        """POST /api/v1/cognify — build the knowledge graph."""
        session = await self._get_session()

        payload: dict = {}
        if datasets:
            payload["datasets"] = (
                [str(d) for d in datasets] if isinstance(datasets, list) else [str(datasets)]
            )
        if kwargs.get("run_in_background"):
            payload["run_in_background"] = True
        if kwargs.get("custom_prompt"):
            payload["custom_prompt"] = kwargs["custom_prompt"]
        if kwargs.get("chunk_size") is not None:
            payload["chunk_size"] = kwargs["chunk_size"]
        if kwargs.get("chunks_per_batch") is not None:
            payload["chunks_per_batch"] = kwargs["chunks_per_batch"]

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
        if kwargs.get("dataset_ids"):
            dataset_ids = kwargs["dataset_ids"]
            if isinstance(dataset_ids, UUID):
                dataset_ids = [dataset_ids]
            payload["datasetIds"] = [str(dataset_id) for dataset_id in dataset_ids]
        if kwargs.get("top_k") is not None:
            payload["topK"] = kwargs["top_k"]
        if kwargs.get("system_prompt"):
            payload["systemPrompt"] = kwargs["system_prompt"]
        if kwargs.get("node_name"):
            payload["nodeName"] = kwargs["node_name"]
        if kwargs.get("only_context") is not None:
            payload["onlyContext"] = kwargs["only_context"]
        if ContextFormat.parse(kwargs.get("context_format")) is ContextFormat.PROMPT:
            payload["contextFormat"] = ContextFormat.PROMPT.value
        if kwargs.get("verbose") is not None:
            payload["verbose"] = kwargs["verbose"]
        if kwargs.get("skills") is not None:
            payload["skills"] = [
                skill.name if hasattr(skill, "name") else str(skill) for skill in kwargs["skills"]
            ]
        if kwargs.get("tools") is not None:
            payload["tools"] = kwargs["tools"]
        if kwargs.get("max_iter") is not None:
            payload["maxIter"] = kwargs["max_iter"]
        if kwargs.get("include_references") is not None:
            payload["includeReferences"] = kwargs["include_references"]
        if kwargs.get("code_query") is not None:
            payload["codeQuery"] = kwargs["code_query"]

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
            payload["dataset"] = str(kwargs["dataset"])
        if kwargs.get("dataset_id"):
            payload["dataset_id"] = str(kwargs["dataset_id"])
        if kwargs.get("data_id"):
            payload["data_id"] = str(kwargs["data_id"])
        if kwargs.get("memory_only") is not None:
            payload["memory_only"] = bool(kwargs["memory_only"])

        async with session.post(
            f"{self.service_url}/api/v1/forget",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote forget failed ({resp.status}): {body}")
            return await resp.json()
