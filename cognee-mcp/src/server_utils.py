"""Pure helpers for Cognee MCP tool input validation and result rendering."""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_TOP_K = 100
COMPLETION_SEARCH_TYPES = {
    "GRAPH_COMPLETION",
    "GRAPH_COMPLETION_COT",
    "HYBRID_COMPLETION",
    "RAG_COMPLETION",
}
VALID_DELETE_MODES = {"soft", "hard"}


@dataclass(frozen=True)
class ParsedCognifyData:
    items: list[str]
    is_batch: bool


def looks_like_file_path(data: str) -> bool:
    """Return True when a string appears to be a local file path."""
    data = data.strip()
    return data.startswith(("/", "file://")) or bool(re.match(r"^[A-Za-z]:\\", data))


def validate_file_path(
    data: str,
    *,
    path_exists: Callable[[str], bool] = os.path.exists,
    is_running_in_docker: Callable[[], bool] = lambda: False,
) -> str | None:
    """Validate path-like input and return an MCP-friendly error when invalid."""
    if not looks_like_file_path(data):
        return None

    path = data.strip()
    path = path.removeprefix("file://")

    if path_exists(path):
        return None

    msg = f"File not found: {path}"
    if is_running_in_docker():
        msg += (
            "\n\nIt looks like you're running inside Docker. Host file paths are not "
            "accessible inside the container. To ingest local files, mount a volume in "
            "docker-compose.yml:\n"
            "  volumes:\n"
            "    - /path/to/your/data:/data\n"
            "Then reference the file as /data/<filename> instead."
        )
    return msg


def parse_csv_list(value: str | None) -> list[str] | None:
    """Parse an optional comma-separated string into a clean list."""
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def validate_top_k(top_k: int, *, maximum: int = MAX_TOP_K) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer.") from exc

    if value < 1:
        raise ValueError("top_k must be at least 1.")
    if value > maximum:
        raise ValueError(f"top_k must be less than or equal to {maximum}.")
    return value


def normalize_delete_mode(mode: str) -> str:
    normalized = (mode or "soft").strip().lower()
    if normalized not in VALID_DELETE_MODES:
        raise ValueError("mode must be either 'soft' or 'hard'.")
    return normalized


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    return value


def _json_dumps(value: Any, *, json_encoder: type[json.JSONEncoder] | None = None) -> str:
    if json_encoder:
        return json.dumps(_model_dump(value), indent=2, cls=json_encoder)
    return json.dumps(_model_dump(value), indent=2, default=str)


def _get_field(value: Any, *names: str) -> Any:
    value = _model_dump(value)
    if isinstance(value, dict):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            field = getattr(value, name)
            if field is not None:
                return field
    return None


def _unwrap_results(value: Any) -> Any:
    value = _model_dump(value)
    if isinstance(value, dict) and "results" in value:
        return value["results"]
    return value


def _render_scalar_or_json(
    value: Any, *, json_encoder: type[json.JSONEncoder] | None = None
) -> str:
    value = _model_dump(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value, json_encoder=json_encoder)
    return str(value)


def _extract_result_text(value: Any, *, json_encoder: type[json.JSONEncoder] | None = None) -> str:
    text = _get_field(
        value,
        "search_result",
        "result",
        "text",
        "answer",
        "content",
        "completion",
        "text_result",
    )
    if text is None:
        return _render_scalar_or_json(value, json_encoder=json_encoder)
    return _render_scalar_or_json(text, json_encoder=json_encoder)


def _format_completion_results(
    results: Any, *, json_encoder: type[json.JSONEncoder] | None = None
) -> str:
    results = _unwrap_results(results)
    if not isinstance(results, list):
        return _extract_result_text(results, json_encoder=json_encoder)

    lines: list[str] = []
    for result in results:
        dataset_name = _get_field(result, "dataset_name") or "unknown"
        content = _get_field(
            result,
            "search_result",
            "result",
            "text",
            "answer",
            "content",
            "completion",
            "text_result",
        )
        if content is None:
            lines.append(_render_scalar_or_json(result, json_encoder=json_encoder))
            continue

        content_items = content if isinstance(content, list) else [content]
        for item in content_items:
            rendered = _render_scalar_or_json(item, json_encoder=json_encoder)
            prefix = f"[{dataset_name}] " if dataset_name else ""
            lines.append(f"{prefix}{rendered}")

    return (
        "\n\n".join(lines) if lines else _render_scalar_or_json(results, json_encoder=json_encoder)
    )


MAX_RECALL_REFS = 5


def _evidence_rank(reference: Any) -> tuple[bool, int]:
    rank = reference.get("rank") if isinstance(reference, dict) else None
    if isinstance(rank, int) and not isinstance(rank, bool):
        return (False, rank)
    return (True, 0)


def recall_item_dataset(item: Any) -> str | None:
    """The dataset line for a hit whose data ids are cited in its text, not in refs.

    The default hybrid answer lists its sources' data_ids in an ``Evidence:``
    block inside the answer; ``forget`` also needs the dataset, which only the
    hit itself carries.
    """
    dataset_id = _get_field(item, "dataset_id")
    if not dataset_id:
        return None
    name = _get_field(item, "dataset_name")
    if name:
        return f"dataset: {' '.join(str(name).split())[:60]} (dataset_id={dataset_id})"
    return f"dataset: dataset_id={dataset_id}"


def recall_item_refs(item: Any) -> list[dict[str, str]]:
    """Forgettable data items behind one recall hit, as (data_id, dataset_id) pairs.

    Two places carry them: a chunk/summary hit names its own document in
    ``metadata.data_id``, and a completion hit lists the segments it drew on in
    ``metadata.evidence`` (only with ``include_references``). Graph-node, edge
    and triplet references name no document and are skipped. A reference
    without its own ``dataset_id`` takes the hit's, since ``forget`` needs both.
    Evidence is taken best match first (lowest ``rank``; unranked last), so the
    cap keeps the most relevant items.
    """
    metadata = _model_dump(_get_field(item, "metadata"))
    if not isinstance(metadata, dict):
        return []
    hit_dataset_id = _get_field(item, "dataset_id")
    evidence = [_model_dump(reference) for reference in metadata.get("evidence") or []]
    evidence.sort(key=_evidence_rank)
    candidates = [metadata, *evidence]

    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        candidate = _model_dump(candidate)
        if not isinstance(candidate, dict):
            continue
        data_id = candidate.get("data_id")
        dataset_id = candidate.get("dataset_id") or hit_dataset_id
        if not data_id or not dataset_id:
            continue
        key = (str(data_id), str(dataset_id))
        if key in seen:
            continue
        seen.add(key)
        ref = {"data_id": key[0], "dataset_id": key[1]}
        if candidate.get("document_name"):
            ref["document"] = " ".join(str(candidate["document_name"]).split())[:80]
        refs.append(ref)
        if len(refs) == MAX_RECALL_REFS:
            break
    return refs


def format_recall_body(
    results: Any,
    *,
    json_encoder: type[json.JSONEncoder] | None = None,
    include_references: bool = False,
) -> str:
    """Render recall results, including normalized response envelopes.

    With ``include_references``, each hit backed by stored data gets one
    ``refs:`` line per data item, carrying the ids ``forget`` takes. A hit with
    no structured refs (the default hybrid answer cites data_ids in its text)
    gets one ``dataset:`` line instead, so the cited ids can be paired with it.
    """
    results = _unwrap_results(results)
    if not results:
        return "No relevant results found."

    result_items = results if isinstance(results, list) else [results]
    lines: list[str] = []
    for result in result_items:
        source = _get_field(result, "_source", "source")
        text = _get_field(result, "answer", "text", "content", "search_result", "result")
        rendered = _render_scalar_or_json(
            text if text is not None else result,
            json_encoder=json_encoder,
        )
        prefix = f"[{source}] " if source else ""
        line = f"{prefix}{rendered}"
        if include_references:
            refs = recall_item_refs(result)
            for ref in refs:
                line += "\n  refs: " + ", ".join(f"{key}={value}" for key, value in ref.items())
            dataset = None if refs else recall_item_dataset(result)
            if dataset:
                line += f"\n  {dataset}"
        lines.append(line)

    return "\n\n".join(lines)


@dataclass(frozen=True)
class RecallState:
    state: str
    completed: int | None = None
    total: int | None = None


def recall_items(results: Any) -> list[Any]:
    """Count returned entries, not requested top_k, graph nodes, or system markers."""
    results = _unwrap_results(results)
    items = results if isinstance(results, list) else [results]
    return [
        item
        for item in items
        if item is not None
        and item != ""
        and item != {}
        and item != []
        and _get_field(item, "_source", "source") != "system"
    ]


def recall_marker_state(results: Any) -> RecallState | None:
    results = _unwrap_results(results)
    for item in results if isinstance(results, list) else [results]:
        if (
            _get_field(item, "_source", "source") == "system"
            and _get_field(item, "status") == "build_failed"
        ):
            return RecallState("build_failed")
    return None


def classify_recall_state(progress: dict) -> RecallState:
    """Classify authorized datasets from an existing pipeline-status response.

    Three outcomes, because a consumer has three actions: retry shortly
    (``indexing``), go look at why ingestion failed (``build_failed``), or
    accept that there is nothing (``none``).

    Earlier this also separated ``empty`` / ``not_indexed`` / ``no_match`` /
    ``unknown``. Those four produced near-identical sentences, and telling them
    apart was the entire reason for a ``/datasets/graph-summary`` round trip per
    empty recall -- plus, locally, a full per-dataset graph traversal. The
    distinction was unreliable anyway: a zero count from an unavailable graph
    store is not proof of emptiness, so several branches fell back to
    ``unknown`` regardless.
    """
    runs = []
    for value in progress.values():
        if "status" in value:
            runs.append(value)
        else:
            runs.extend(value.values())
    active = [
        run
        for run in runs
        if run.get("status")
        in (
            "DATASET_PROCESSING_STARTED",
            "DATASET_PROCESSING_INITIATED",
        )
    ]
    if active:
        tick = (active[0].get("progress") or {}) if len(active) == 1 else {}
        completed, total = tick.get("completed_items"), tick.get("total_items")
        if type(completed) is int and type(total) is int and 0 <= completed <= total and total > 0:
            return RecallState("indexing", completed, total)
        return RecallState("indexing")
    if any(run.get("status") == "DATASET_PROCESSING_ERRORED" for run in runs):
        return RecallState("build_failed")
    return RecallState("none")


def format_recall_results(
    results: Any,
    *,
    json_encoder: type[json.JSONEncoder] | None = None,
    empty_state: RecallState | None = None,
    items: list[Any] | None = None,
    include_references: bool = False,
) -> str:
    """Add one summary line; preserve the existing body beneath it.

    `items` lets a caller that has already extracted them pass them in.
    recall() computes them to decide whether to run diagnostics at all, and
    re-deriving them here walked the same payload a second time.
    """
    items = recall_items(results) if items is None else items
    count = len(items)
    if count:
        summary = f"{count} {'memory' if count == 1 else 'memories'} found"
        sources: dict[str, int] = {}
        for item in items:
            dataset = _get_field(item, "dataset_name")
            source = _get_field(item, "_source", "source")
            hint = dataset or {"session": "sessions", "graph": "graph"}.get(source, source)
            if hint:
                # Metadata must not create another summary line or an unbounded header.
                hint = " ".join(str(hint).split())[:60]
                sources[hint] = sources.get(hint, 0) + 1
        if sources:
            hints = [f"{n} from {hint}" for hint, n in list(sources.items())[:3]]
            summary += " (" + ", ".join(hints) + ")"
    else:
        state = empty_state or recall_marker_state(results) or RecallState("none")
        summary = {
            "indexing": "still indexing — retry shortly",
            "build_failed": "memory indexing failed — check cognify_status",
            "none": "no matching memories",
        }.get(state.state, "no matching memories")
        if state.state == "indexing" and state.total is not None:
            summary = (
                f"still indexing — {state.completed}/{state.total} items processed, retry shortly"
            )
    body = (
        format_recall_body(
            results, json_encoder=json_encoder, include_references=include_references
        )
        if _unwrap_results(results)
        else ""
    )
    return f"{summary}\n{body}" if body else summary
