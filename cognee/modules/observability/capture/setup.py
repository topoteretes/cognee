"""The setup that produced a run's events (SDK-529).

An eval that compares extraction or retrieval quality needs to know what
configuration produced each run: which LLM and which embedding model (hosted
or local), how structured output was obtained, which graph extractor ran — the
LLM path or the local GLiNER demo model — and which database backends were
behind it. ``capture_setup()`` snapshots exactly that, and
``setup_fingerprint()`` reduces it to one stable id, so runs can be bucketed
by configuration without diffing dicts. Every run manifest carries both
(``setup``, ``setup_fingerprint``); a ``setup.changed`` event is emitted the
first time a process observes a setup for a run kind and whenever a later
run's fingerprint differs from the previous one of that kind, carrying the
per-key diff — the chronological marker for "the model changed here".

The snapshot is process-level configuration at run start. The exact model a
stage then used is recorded by the stage itself on the manifest
(``extraction.extractor`` / ``extraction.model`` — the GLiNER demo's local
model id included — and ``summarization.model``), because a per-call override
can differ from the process default.

Secret-free by construction: the values come from ``get_current_settings()``
(the provider stack telemetry already reports) through an ALLOWLIST of keys,
never the whole dict — that dict carries database URLs, which can embed
credentials and are only safe for telemetry because its sanitizer fingerprints
them. A model that is a filesystem path is reported as ``local_path`` (same as
telemetry); a hosted or local-server model id (``llama3.1:8b``,
``BAAI/bge-small-en-v1.5``) is kept, since the model identity is the point.
Nothing here raises: a snapshot that cannot be taken is ``None`` and the
manifest says so.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# (manifest key, path into get_current_settings()' dict). Adding a key here is
# additive for consumers and changes every fingerprint; removing or renaming one
# is a CAPTURE_SCHEMA_VERSION bump.
SETUP_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("llm.provider", ("llm", "provider")),
    ("llm.model", ("llm", "model")),
    # Whether that LLM is usable at all (a keyless install reports the unused
    # default model with configured=False and the GLiNER extractor).
    ("llm.configured", ("llm", "configured")),
    ("llm.structured_output", ("llm", "structured_output")),
    ("llm.instructor_mode", ("llm", "instructor_mode")),
    ("embedding.provider", ("embedding", "provider")),
    ("embedding.model", ("embedding", "model")),
    # "llm" or "gliner_demo" for a cognify run; None for a run that extracts no
    # graph (a search), so the two kinds never trip each other's setup.changed.
    ("graph_extractor", ("graph_extractor",)),
    ("graph.provider", ("graph", "provider")),
    ("vector.provider", ("vector", "provider")),
    ("relational.provider", ("relational", "provider")),
)

_SCALARS = (str, int, float, bool)


def _lookup(settings: Any, path: tuple[str, ...]) -> Any:
    value = settings
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if value is None or isinstance(value, _SCALARS):
        return value
    return str(value)


def capture_setup(*, graph_extractor: str | None = None) -> dict[str, Any] | None:
    """Snapshot the allowlisted setup fields; ``None`` if the snapshot cannot be taken.

    ``graph_extractor`` is cognify's already-resolved extractor for a pipeline
    run; leave it ``None`` for runs that extract no graph (the field is then
    ``None``, not telemetry's ``"unknown"``, so a search and a cognify on the
    same configuration differ only in that one key).
    """
    try:
        # Lazy: keeps this package free of the settings/config import chain, and
        # ``import cognee`` free of this package.
        from cognee.modules.settings.get_current_settings import get_current_settings

        settings = get_current_settings(graph_extractor=graph_extractor)
    except Exception:  # a snapshot never breaks the run it describes
        return None

    setup = {key: _lookup(settings, path) for key, path in SETUP_FIELDS}
    if graph_extractor is None:
        setup["graph_extractor"] = None
    return setup


def setup_fingerprint(setup: dict[str, Any] | None) -> str | None:
    """``sha256:<16 hex>`` over the canonical JSON of the setup; key order irrelevant."""
    if setup is None:
        return None
    canonical = json.dumps(setup, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def setup_diff(
    previous: dict[str, Any] | None, current: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """``{key: {"from": old, "to": new}}`` for every key whose value differs."""
    previous = previous or {}
    current = current or {}
    return {
        key: {"from": previous.get(key), "to": current.get(key)}
        for key in sorted(set(previous) | set(current))
        if previous.get(key) != current.get(key)
    }
