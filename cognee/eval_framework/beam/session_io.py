"""Scale-agnostic ingestion-ready BEAM session-JSON contract.

One implementation of the read/parse/write primitives that both the local and Modal
ingestion paths need for BEAM's JSON-list session files, previously duplicated (with a
``_``-prefixed naming convention on the Modal side) between
``beam_ingest_conversation_sessions_with_distillation.py`` and
``beam_json_sessions_modal.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return payload


def safe_slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "none"


def session_id_for(dataset_name: str, source_path: Path) -> str:
    return f"beam_{safe_slug(dataset_name)}_{safe_slug(source_path.stem)}"


def parse_beam_block(block: str) -> dict[str, str] | None:
    """Parse a BEAM turn block into a ``{user, assistant}`` pair."""
    user_at = block.find("User:")
    assistant_at = block.find("Assistant:")
    if user_at == -1 or assistant_at == -1 or assistant_at < user_at:
        return None
    user = block[user_at + len("User:") : assistant_at].strip()
    assistant = block[assistant_at + len("Assistant:") :].strip()
    return {"user": user, "assistant": assistant} if user else None


def parse_turns(path: Path) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for item in read_json_list(path):
        if isinstance(item, str):
            parsed = parse_beam_block(item)
            if parsed is not None:
                turns.append(parsed)
            continue
        if isinstance(item, dict) and item.get("user"):
            turns.append({"user": str(item["user"]), "assistant": str(item.get("assistant", ""))})
    return turns


def distillation_report_payload(result: Any, seconds: float) -> dict[str, Any]:
    documents = getattr(result, "documents", None) or []
    return {
        "status": getattr(result, "status", None),
        "gated_entry_count": getattr(result, "gated_entry_count", None),
        "batch_count": getattr(result, "batch_count", None),
        "proposed_lesson_count": getattr(result, "proposed_lesson_count", None),
        "accepted_lesson_count": getattr(result, "accepted_lesson_count", None),
        "rejected_lesson_count": getattr(result, "rejected_lesson_count", None),
        "document_count": len(documents),
        "document_lengths": [len(document) for document in documents],
        "document_length": sum(len(document) for document in documents),
        "distillation_seconds": seconds,
    }
