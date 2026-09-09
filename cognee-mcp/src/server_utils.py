"""Pure helpers for Cognee MCP tool input validation and result rendering."""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


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
    return (
        data.startswith("/") or bool(re.match(r"^[A-Za-z]:\\", data)) or data.startswith("file://")
    )


def validate_file_path(
    data: str,
    *,
    path_exists: Callable[[str], bool] = os.path.exists,
    is_running_in_docker: Callable[[], bool] = lambda: False,
) -> Optional[str]:
    """Validate path-like input and return an MCP-friendly error when invalid."""
    if not looks_like_file_path(data):
        return None

    path = data.strip()
    if path.startswith("file://"):
        path = path[7:]

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


def parse_csv_list(value: Optional[str]) -> Optional[list[str]]:
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


def _json_dumps(value: Any, *, json_encoder: Optional[type[json.JSONEncoder]] = None) -> str:
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
    value: Any, *, json_encoder: Optional[type[json.JSONEncoder]] = None
) -> str:
    value = _model_dump(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value, json_encoder=json_encoder)
    return str(value)


def _extract_result_text(
    value: Any, *, json_encoder: Optional[type[json.JSONEncoder]] = None
) -> str:
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
    results: Any, *, json_encoder: Optional[type[json.JSONEncoder]] = None
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


def format_recall_results(
    results: Any, *, json_encoder: Optional[type[json.JSONEncoder]] = None
) -> str:
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
