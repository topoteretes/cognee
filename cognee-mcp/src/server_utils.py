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


def format_recall_body(results: Any, *, json_encoder: type[json.JSONEncoder] | None = None) -> str:
    """Render recall results, including normalized response envelopes."""
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
        lines.append(f"{prefix}{rendered}")

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


# recall()'s system markers while memory is searchable but its graph is not
# built yet (status ``memory_<state>``), folded to the two words a consumer
# acts on: wait a little, or accept chunk-level answers for now.
_PARTIAL_MARKER_STATES = {
    "memory_caching": "indexing",
    "memory_indexing": "indexing",
    "memory_extracting": "partial",
    "memory_cached": "partial",
    "memory_indexed": "partial",
    "memory_caching_failed": "partial",
    "memory_indexing_failed": "partial",
    "memory_extraction_failed": "partial",
}


def recall_marker_state(results: Any) -> RecallState | None:
    results = _unwrap_results(results)
    for item in results if isinstance(results, list) else [results]:
        if _get_field(item, "_source", "source") != "system":
            continue
        status = _get_field(item, "status")
        if status == "build_failed":
            return RecallState("build_failed")
        if status in _PARTIAL_MARKER_STATES:
            return RecallState(_PARTIAL_MARKER_STATES[status])
    return None


# Preparation runs of cognify's staged build (SDK-397) and the runs that build
# the graph. A preparation run that failed is no build failure once a build
# completed for that dataset: the graph answers anyway.
_PREPARATION_PIPELINES = ("caching_pipeline", "indexing_pipeline")
_GRAPH_BUILD_PIPELINES = ("cognify_pipeline", "code_graph_pipeline")


def _run_status(run: dict) -> Any:
    """A run's status as its string value: the API sends strings, direct mode enums."""
    status = run.get("status")
    return getattr(status, "value", status)


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

    A legacy ``DATASET_PROCESSING_INITIATED`` row is not a run in progress (it
    is no longer written), and an errored caching or indexing run does not
    make a dataset whose graph build completed ``build_failed``.
    """
    runs: list[tuple[str, str | None, dict]] = []
    for dataset, value in progress.items():
        if "status" in value:
            runs.append((dataset, None, value))
        else:
            runs.extend((dataset, pipeline, run) for pipeline, run in value.items())
    active = [run for _, _, run in runs if _run_status(run) == "DATASET_PROCESSING_STARTED"]
    if active:
        tick = (active[0].get("progress") or {}) if len(active) == 1 else {}
        completed, total = tick.get("completed_items"), tick.get("total_items")
        if type(completed) is int and type(total) is int and 0 <= completed <= total and total > 0:
            return RecallState("indexing", completed, total)
        return RecallState("indexing")
    built = {
        dataset
        for dataset, pipeline, run in runs
        if pipeline in _GRAPH_BUILD_PIPELINES and _run_status(run) == "DATASET_PROCESSING_COMPLETED"
    }
    if any(
        _run_status(run) == "DATASET_PROCESSING_ERRORED"
        and not (pipeline in _PREPARATION_PIPELINES and dataset in built)
        for dataset, pipeline, run in runs
    ):
        return RecallState("build_failed")
    return RecallState("none")


def format_recall_results(
    results: Any,
    *,
    json_encoder: type[json.JSONEncoder] | None = None,
    empty_state: RecallState | None = None,
    items: list[Any] | None = None,
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
        marker = recall_marker_state(results)
        if marker is not None and marker.state == "partial":
            summary += " — early results: the knowledge graph is still being built"
    else:
        state = empty_state or recall_marker_state(results) or RecallState("none")
        summary = {
            "indexing": "still indexing — retry shortly",
            "partial": "no matching memories yet — the graph is still being built, retry later",
            "build_failed": "memory indexing failed — check cognify_status",
            "none": "no matching memories",
        }.get(state.state, "no matching memories")
        if state.state == "indexing" and state.total is not None:
            summary = (
                f"still indexing — {state.completed}/{state.total} items processed, retry shortly"
            )
    body = (
        format_recall_body(results, json_encoder=json_encoder) if _unwrap_results(results) else ""
    )
    return f"{summary}\n{body}" if body else summary
