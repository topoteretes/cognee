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


def recall_marker_state(results: Any) -> RecallState | None:
    results = _unwrap_results(results)
    for item in results if isinstance(results, list) else [results]:
        if (
            _get_field(item, "_source", "source") == "system"
            and _get_field(item, "status") == "build_failed"
        ):
            return RecallState("build_failed")
    return None


def classify_recall_state(progress: dict, graphs: list[dict]) -> RecallState:
    """Classify authorized datasets from existing status and graph-summary responses."""
    runs = []
    graph_runs = []
    for value in progress.values():
        if "status" in value:
            runs.append(value)
            graph_runs.append(value)
        else:
            runs.extend(value.values())
            graph_runs.extend(run for pipeline, run in value.items() if pipeline != "add_pipeline")
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
    if not graphs:
        return RecallState("unknown")
    if any(graph.get("num_nodes", graph.get("numNodes", 0)) > 0 for graph in graphs):
        return RecallState("no_match")
    # A zero from an unavailable graph store is not proof that memory is empty.
    if any(
        graph.get("pipeline_run_id", graph.get("pipelineRunId")) is not None
        and graph.get("computed_at", graph.get("computedAt")) is None
        for graph in graphs
    ):
        return RecallState("unknown")
    if all(graph.get("computed_at", graph.get("computedAt")) is not None for graph in graphs):
        return RecallState("empty")
    # Graph summaries cover cognify, not arbitrary/custom or code pipelines.
    # A completed run with no corresponding graph count must not look empty.
    if any(run.get("status") == "DATASET_PROCESSING_COMPLETED" for run in graph_runs):
        return RecallState("no_match")
    return RecallState("not_indexed")


def format_recall_results(
    results: Any,
    *,
    json_encoder: type[json.JSONEncoder] | None = None,
    empty_state: RecallState | None = None,
) -> str:
    """Add one summary line; preserve the existing body beneath it."""
    items = recall_items(results)
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
        state = empty_state or recall_marker_state(results) or RecallState("unknown")
        summary = {
            "empty": "memory graph is empty — no indexed memories available",
            "not_indexed": "memory has not been indexed yet — add data and run indexing",
            "no_match": "no matching memories",
            "indexing": "still indexing — retry shortly",
            "build_failed": "memory indexing failed — check cognify_status",
            "unknown": "no matching memories returned — memory status unavailable",
        }.get(state.state, "no matching memories returned — memory status unavailable")
        if state.state == "indexing" and state.total is not None:
            summary = (
                f"still indexing — {state.completed}/{state.total} items processed, retry shortly"
            )
    body = (
        format_recall_body(results, json_encoder=json_encoder) if _unwrap_results(results) else ""
    )
    return f"{summary}\n{body}" if body else summary
