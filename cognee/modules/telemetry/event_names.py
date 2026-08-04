"""Normalise telemetry event names into a stable contract for consumers.

Event names grew organically into five incompatible styles — measured across
three days of production traffic there are 60+ distinct names:

===============================================  ===========================
style                                            examples
===============================================  ===========================
``cognee.<op>``                                  ``cognee.remember``
``cognee.<op> EXECUTION <STATE>``                ``cognee.search EXECUTION STARTED``
``<Thing> API Endpoint Invoked``                 ``Remember Entry API Endpoint Invoked``
``Pipeline Run <State>``                         ``Pipeline Run Completed``
``<Type> Task <State>``                          ``Coroutine Task Started``
===============================================  ===========================

Renaming the emitters would give one clean scheme, but the same names are the
warehouse's ``pipeline_events.tracking_event`` — renaming breaks every existing
dashboard and splits history at the cutover. So the raw name is preserved as
``event_name`` and this module derives two stable fields alongside it:

``operation``
    The user-facing verb: ``remember``, ``recall``, ``cognify``, … A UI groups
    and filters on this instead of pattern-matching names. ``None`` for events
    that aren't tied to one operation (pipeline and task bookkeeping).

``event_kind``
    Which layer emitted it, so a consumer can pick one and avoid showing the
    same action several times:

    - ``operation`` — the SDK-level call a user made
    - ``endpoint``  — an HTTP route was invoked
    - ``pipeline``  — a pipeline run started/finished
    - ``task``      — internal task bookkeeping (excluded from local sinks)

Both are derived, so adding an emitter needs no change here: it falls through to
the heuristics and still gets a sensible ``operation``/``event_kind``.
"""

import re

TASK_SUFFIXES = (" Task Started", " Task Completed", " Task Errored")
_ENDPOINT_SUFFIX = " API Endpoint Invoked"
# The `cognee.` prefix is optional: some emitters use it, some don't
# (`code_description_to_code_part_search EXECUTION STARTED`).
_EXECUTION_RE = re.compile(r"^(?:cognee\.)?(?P<op>[\w.]+) EXECUTION [A-Z]+$")

KIND_OPERATION = "operation"
KIND_ENDPOINT = "endpoint"
KIND_PIPELINE = "pipeline"
KIND_TASK = "task"

# Endpoint prefixes whose first word alone would read badly or split one
# operation across two names. Everything else falls back to its first word.
_ENDPOINT_OVERRIDES = {
    "Api Key Management": "api_key",
    "Add By Text": "add",
    "Add AISOC": "add",
    "Cognify AIPTS": "cognify",
    "Cognify AISOC": "cognify",
    "Remember Entry": "remember",
    "Sync Status Overview": "sync",
    "Visualize Live Events": "visualize",
    "Visualize Brains": "visualize",
    "Schema Inventory": "schema",
    "Schema Provenance JSON": "schema",
    "Skills List": "skills",
    "Skill Ingest": "skills",
    "Skill Proposal Get": "proposals",
    "Ontology Upload": "ontology",
    "Ontology Delete": "ontology",
    "Ontology List": "ontology",
    "List Principal Dataset Grants": "permissions",
    "Knowledge Views": "knowledge_views",
    "LLM Custom Prompt Endpoint": "llm",
    "LLM Infer Schema Endpoint": "llm",
    "Cloud Sync": "sync",
}


def _slug(value: str) -> str:
    """Lower-case, underscore-separated, punctuation stripped."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def normalize_event(event_name: str) -> tuple[str | None, str]:
    """Return ``(operation, event_kind)`` for a raw telemetry event name.

    Never raises and never returns an empty ``event_kind`` — an unrecognised
    name is treated as an operation named after itself, which is strictly more
    useful to a consumer than a null.
    """
    if not event_name:
        return None, KIND_OPERATION

    # Internal bookkeeping: no single operation owns these.
    if event_name.endswith(TASK_SUFFIXES):
        return None, KIND_TASK
    if event_name.startswith("Pipeline Run "):
        return None, KIND_PIPELINE

    if event_name.endswith(_ENDPOINT_SUFFIX):
        prefix = event_name[: -len(_ENDPOINT_SUFFIX)].strip()
        # Some prefixes already end in "Endpoint" (LLM Custom Prompt Endpoint).
        operation = _ENDPOINT_OVERRIDES.get(prefix)
        if operation is None:
            operation = _slug(prefix.split()[0]) if prefix else None
        return operation, KIND_ENDPOINT

    execution = _EXECUTION_RE.match(event_name)
    if execution:
        return _slug(execution.group("op").split(".")[0]), KIND_OPERATION

    if event_name.startswith("cognee."):
        # cognee.remember -> remember; cognee.session.add_qa -> session;
        # cognee.remember.import -> remember; and drop any trailing prose, as in
        # "cognee.cognify DEFAULT TASKS CREATION ERRORED" -> cognify.
        tail = event_name[len("cognee.") :].split()[0]
        return _slug(tail.split(".")[0]), KIND_OPERATION

    return _slug(event_name) or None, KIND_OPERATION
