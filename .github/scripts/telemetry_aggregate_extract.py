"""Extract ANONYMIZED AGGREGATES from the MotherDuck telemetry warehouse.

This script is the privacy boundary of the daily telemetry-insights Action:
the analysis model (Claude Code) never receives warehouse credentials and
never sees a raw event — only the CSV aggregates this script emits.

Hard rules enforced here:
- Only the queries below run; every SELECT lists explicit output columns.
- Free-text / PII-bearing fields are NEVER selected: search_query,
  system_prompt, dataset names, raw properties, tenant ids, endpoints'
  query strings, error text. Error events contribute only Python class names
  (``exception_type``, ``exception_cause``), allowlisted to identifier
  characters, and an HTTP ``status_code``.
- Identity columns (user_id, api_key_hash, anonymous_id, persistent_id)
  are used ONLY inside COUNT(DISTINCT ...); their values are never emitted.
- Pipeline run ids are used ONLY to group a run's events and inside
  COUNT(DISTINCT ...); their values are never emitted.
- Identifier-bearing provider/model settings are bucketed as 'redacted'
  before grouping, so custom deployment names cannot stop the daily export.
- A post-write guard fails the job if any output header matches the
  denylist or any cell matches identifier patterns (email, UUID, ak_ hash).

Output: telemetry_aggregates/*.csv covering the last WINDOW_DAYS days
(default 70, so the analyzer can compute week-over-week and month-over-month
comparisons inside the window).
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

import duckdb

WINDOW_DAYS = int(os.getenv("TELEMETRY_WINDOW_DAYS", "70"))
TASK_WINDOW_DAYS = min(WINDOW_DAYS, 14)
OUT_DIR = Path(os.getenv("TELEMETRY_OUT_DIR", "telemetry_aggregates"))

# Shared by SQL redaction and the independent post-write guard.
CELL_PATTERNS = (
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),  # email
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),  # uuid
    re.compile(r"\bak_[0-9a-f]{16,}\b"),  # key hash
    # A filesystem path with an account directory or a drive letter: a model or
    # provider setting pointing at a local file names the OS account.
    re.compile(r"(^|[^A-Za-z0-9])(/Users/|/home/|/root/|[A-Za-z]:\\)[^\s,;]*"),
)
# The SQL redaction runs on lower-cased values, so it also needs the lower-case
# macOS home directory; the post-write guard keeps the exact case, because route
# templates legitimately contain "/users/".
_SQL_CELL_PATTERN = "|".join(
    [pattern.pattern for pattern in CELL_PATTERNS] + [r"(^|[^a-z0-9])/users/[^\s,;]*"]
).replace("'", "''")


def _provider_dimension(property_path: str, *, max_length: int | None = None) -> str:
    """Bucket identifiers in a fixed provider/model property before aggregation."""
    value = f"json_extract_string(properties, '$.{property_path}')"
    identifier_check = f"regexp_matches(lower({value}), '{_SQL_CELL_PATTERN}')"
    output = value
    if max_length is not None:
        output = f"left(lower({value}), {max_length})"
        # Truncation can hide an identifier or create a word boundary that makes
        # the shortened value match the guard. Inspect both full and output forms.
        identifier_check += f" OR regexp_matches({output}, '{_SQL_CELL_PATTERN}')"
    return f"CASE WHEN {identifier_check} THEN 'redacted' ELSE {output} END"


# Events worth analyzing; everything else (internal task/coroutine spam) is skipped.
EVENT_ALLOWLIST = (
    "cognee.search EXECUTION STARTED",
    "cognee.search EXECUTION COMPLETED",
    "cognee.search EXECUTION ERRORED",
    "cognee.recall",
    "cognee.recall ERRORED",
    "cognee.improve",
    "cognee.forget",
    "cognee.export",
    "cognee.push",
    "cognee.remember.import",
    "cognee.remember.code_graph",
    "cognee.add EXECUTION STARTED",
    "cognee.add EXECUTION COMPLETED",
    "cognee.cognify EXECUTION STARTED",
    "cognee.cognify EXECUTION COMPLETED",
    "cognee.cognify EXECUTION ERRORED",
    "cognee.remember",
    "cognee.session.add_qa",
    "Search API Endpoint Invoked",
    "Add API Endpoint Invoked",
    "Cognify API Endpoint Invoked",
    "Remember API Endpoint Invoked",
    "Remember Entry API Endpoint Invoked",
    "Recall API Endpoint Invoked",
    "Improve API Endpoint Invoked",
    "Forget API Endpoint Invoked",
    "API Exception Raised",
    "GLiNER Runtime Install Started",
    "GLiNER Runtime Install Completed",
    "GLiNER Runtime Install Failed",
    "Pipeline Run Started",
    "Pipeline Run Completed",
    "Pipeline Run Errored",
    "Pipeline Item Started",
    "Pipeline Item Completed",
    "Pipeline Item Errored",
)

# The pseudonymous deployment identity, in decreasing stability order:
# LLM-key hash (org-stable) -> persistent_id (machine-stable, survives user
# recreation; emitted since ~Apr 2026) -> user_id (recreated per install/job).
# This collapses products that mint a fresh user per agent job, so distinct
# counts approximate deployments rather than throwaway identities.
# Used strictly inside COUNT(DISTINCT ...) — never selected as a column.
_IDENT = (
    "coalesce(nullif(json_extract_string(properties, '$.api_key_hash'), ''), "
    "nullif(json_extract_string(properties, '$.persistent_id'), ''), user_id)"
)
# Route templates. Older builds put the raw path in ``endpoint`` (a dataset or user
# id inside it), so ids are folded back into a placeholder before grouping; the
# guard would otherwise refuse the file, and the id is not a dimension anyway.
_ENDPOINT = (
    "regexp_replace(endpoint, "
    "'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', '{id}', 'g')"
)
# Surface the event came from: 'sdk' (the default), 'api', 'cli', 'mcp' (set by each
# entrypoint since SDK-775), 'cloud' (set by the managed cloud). Safe enum.
_ORIGIN = "coalesce(json_extract_string(properties, '$.telemetry_origin'), 'unknown')"
# Normalized version: strip the -local suffix so builds compare cleanly. The
# missing versions are kept apart: 'unknown-null' is an event without the field,
# 'unknown-recovered' a startup-recovery event (it sends the literal "unknown" on
# purpose, its Started event has the version) and 'unknown-unresolved' a client
# that sent "unknown" because it could not resolve its own version.
_VERSION = (
    "CASE WHEN cognee_version IS NULL THEN 'unknown-null' "
    "WHEN cognee_version = 'unknown' "
    "AND json_extract_string(properties, '$.recovered_at_startup') = 'true' "
    "THEN 'unknown-recovered' "
    "WHEN cognee_version = 'unknown' THEN 'unknown-unresolved' "
    "ELSE regexp_replace(cognee_version, '-local$', '') END"
)
# Whether the LLM was usable (a key set, or a provider that needs none). A keyless
# install still reports the default llm provider/model; this tells the two apart.
# Rows from builds before the field are 'unknown'.
_LLM_CONFIGURED = (
    "CASE json_extract_string(properties, '$.llm.configured') "
    "WHEN 'true' THEN 'true' WHEN 'false' THEN 'false' ELSE 'unknown' END"
)
# A run's random id (``pipeline_run_id``): joins the per-item events of one run.
_RUN_ID = "json_extract_string(properties, '$.pipeline_run_id')"


def _closed_value(
    property_path: str, pattern: str = "^[A-Za-z0-9_.,:-]{1,64}$", missing: str = "unknown"
) -> str:
    """A property that should be one of a small set of values, or a bucket.

    ``missing`` (``unknown`` by default) when the event lacks the field — a
    build before it, or a field sent only when there is something to say —
    ``redacted`` when the value is outside ``pattern`` (a fork, a typo, free
    text), so an unexpected value can neither stop the export nor carry an
    identifier.
    """
    value = f"json_extract_string(properties, '$.{property_path}')"
    return (
        f"CASE WHEN {value} IS NULL THEN '{missing}' "
        f"WHEN regexp_matches({value}, '{pattern}') THEN {value} ELSE 'redacted' END"
    )


# Class names use the same validation/bucketing rule as other closed values.
_EXCEPTION_TYPE = _closed_value("exception_type", "^[A-Za-z_][A-Za-z0-9_]*$")
_EXCEPTION_CAUSE = _closed_value("exception_cause", "^[A-Za-z_][A-Za-z0-9_]*$", missing="none")
_RUN_SCOPE = "coalesce(json_extract_string(properties, '$.pipeline_event_scope') = 'run', false)"
_RECOVERED = "coalesce(json_extract_string(properties, '$.recovered_at_startup') = 'true', false)"
_EVENT_SCOPE = f"CASE WHEN {_RUN_SCOPE} OR {_RECOVERED} THEN 'run' ELSE 'legacy_item' END"

# The HTTP status closest to a failure, when any error in its chain carried one
# (a provider's 429/401/503, a cognee error's mapped status); ``none`` otherwise.
_STATUS_CODE = _closed_value("status_code", "^[1-5][0-9][0-9]$", missing="none")
# The structured-output path of LLM calls: closed values, ``unknown`` before the field.
_STRUCTURED_OUTPUT = _closed_value(
    "llm.structured_output", "^(litellm_native|instructor|baml|invalid|unknown)$"
)
# The data item a pipeline run processed (SDK-775): a loader registry name and
# size/token classes, as ``data_item_telemetry_properties`` labels them.
_ITEM_LOADER = _closed_value("item_loader", "^[a-z0-9_]{1,40}$")
_ITEM_SIZE_BUCKET = _closed_value(
    "item_size_bucket", "^(lt_10kb|10kb_100kb|100kb_1mb|1mb_10mb|10mb_100mb|gt_100mb)$"
)
_ITEM_TOKEN_BUCKET = _closed_value("item_token_bucket", "^(lt_1k|1k_10k|10k_100k|100k_1m|gt_1m)$")


# Task names are cognee function names (``extract_graph_from_data``): identifiers.
_TASK_NAME = (
    "CASE WHEN task_name IS NULL THEN 'unknown' "
    "WHEN regexp_matches(task_name, '^[A-Za-z_][A-Za-z0-9_]{0,80}$') THEN task_name "
    "ELSE 'redacted' END"
)
# Session counts on improve() bucketed, so the CSV carries a size class, not a number
# that could single out a deployment.
_SESSION_BUCKET = (
    "CASE WHEN json_extract_string(properties, '$.session_count') IS NULL THEN 'unknown' "
    "WHEN try_cast(json_extract_string(properties, '$.session_count') AS INTEGER) = 0 THEN '0' "
    "WHEN try_cast(json_extract_string(properties, '$.session_count') AS INTEGER) = 1 THEN '1' "
    "WHEN try_cast(json_extract_string(properties, '$.session_count') AS INTEGER) <= 5 THEN '2-5' "
    "WHEN try_cast(json_extract_string(properties, '$.session_count') AS INTEGER) > 5 THEN '6+' "
    "ELSE 'redacted' END"
)

_EVENTS_SQL = "(" + ",".join(f"'{e}'" for e in EVENT_ALLOWLIST) + ")"
_BASE_FILTER = (
    f"ingestion_date >= current_date - INTERVAL {WINDOW_DAYS} DAY "
    f"AND tracking_event IN {_EVENTS_SQL}"
)

# Both outcome and duration reports use the same authoritative lifecycle.
# Historical Pipeline Run events were item events. Even matching start/end
# counts cannot prove the durable-storage flush succeeded, so never infer a
# successful run from them. Recovery can authoritatively close an older run.
_RUNS_CTE = f"""
    WITH runs AS (
        SELECT {_RUN_ID} AS run_id,
               min(ingestion_date) FILTER (tracking_event = 'Pipeline Run Started') AS day,
               min({_VERSION}) FILTER (tracking_event = 'Pipeline Run Started') AS version,
               bool_or(tracking_event = 'Pipeline Run Started' AND {_RUN_SCOPE}) AS authoritative_start,
               bool_or(tracking_event = 'Pipeline Run Completed' AND {_RUN_SCOPE}) AS completed,
               bool_or(tracking_event = 'Pipeline Run Errored' AND ({_RUN_SCOPE} OR {_RECOVERED})) AS errored,
               min(event_timestamp) FILTER (tracking_event = 'Pipeline Run Started' AND {_RUN_SCOPE}) AS started_at,
               max(event_timestamp) FILTER (
                   tracking_event IN ('Pipeline Run Completed', 'Pipeline Run Errored')
                   AND {_RUN_SCOPE} AND NOT {_RECOVERED}) AS ended_at,
               bool_or({_RECOVERED}) AS recovered
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event LIKE 'Pipeline Run%'
              AND {_RUN_ID} IS NOT NULL
        GROUP BY {_RUN_ID}
    )
"""

QUERIES: dict[str, str] = {
    # Daily volume + reach per event, per surface, per version.
    "daily_event_volumes": f"""
        SELECT ingestion_date AS day, tracking_event, {_VERSION} AS version,
               {_ORIGIN} AS origin,
               (cognee_version LIKE '%-local') AS self_hosted,
               count(*) AS events,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER}
        GROUP BY ALL ORDER BY day, tracking_event
    """,
    # Which error classes end pipeline runs, by day and version (SDK-775), with
    # the cause under the wrapper and the HTTP status closest to the failure.
    # Class names and a status are all an Errored event says about its error.
    "pipeline_error_types_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version, {_EVENT_SCOPE} AS event_scope,
               {_provider_dimension("llm.provider")} AS llm_provider,
               {_provider_dimension("embedding.provider")} AS embedding_provider,
               {_provider_dimension("graph.provider")} AS graph_provider,
               {_provider_dimension("vector.provider")} AS vector_provider,
               {_STRUCTURED_OUTPUT} AS structured_output,
               {_EXCEPTION_TYPE} AS exception_type,
               {_EXCEPTION_CAUSE} AS exception_cause,
               {_STATUS_CODE} AS status_code,
               count(*) AS errors,
               count(DISTINCT {_RUN_ID}) AS runs,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event = 'Pipeline Run Errored'
        GROUP BY ALL ORDER BY day, errors DESC
    """,
    # A whole run ends only at the lifecycle owned by run_tasks(), after flush.
    "pipeline_runs_daily": f"""
        {_RUNS_CTE}
        SELECT day, version,
               count(*) AS runs_started,
               count(*) FILTER (completed AND NOT errored) AS runs_completed,
               count(*) FILTER (errored) AS runs_errored,
               count(*) FILTER (authoritative_start AND NOT completed AND NOT errored) AS runs_silent,
               count(*) FILTER (NOT authoritative_start AND NOT completed AND NOT errored) AS runs_unclassified
        FROM runs
        WHERE day IS NOT NULL
        GROUP BY ALL ORDER BY day, version
    """,
    # Graph-build pipeline health by day and version.
    "pipeline_outcomes_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version, {_EVENT_SCOPE} AS event_scope,
               count(*) FILTER (tracking_event = 'Pipeline Run Started')   AS started,
               count(*) FILTER (tracking_event = 'Pipeline Run Completed') AS completed,
               count(*) FILTER (tracking_event = 'Pipeline Run Errored')   AS errored,
               count(DISTINCT {_IDENT}) FILTER (tracking_event = 'Pipeline Run Errored')
                   AS identities_with_errors
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event LIKE 'Pipeline Run%'
        GROUP BY ALL ORDER BY day, version
    """,
    # Pipeline outcomes by what was ingested (SDK-775): the loader that produced
    # the item's text and its size and token classes, so a failure concentrated
    # in one loader or one size class is visible. Rows from builds before the
    # fields carry ``unknown``; a custom pipeline's items carry no profile.
    "pipeline_item_outcomes_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version,
               {_ITEM_LOADER} AS item_loader,
               {_ITEM_SIZE_BUCKET} AS item_size_bucket,
               {_ITEM_TOKEN_BUCKET} AS item_token_bucket,
               count(*) FILTER (tracking_event IN ('Pipeline Run Started', 'Pipeline Item Started'))   AS started,
               count(*) FILTER (tracking_event IN ('Pipeline Run Completed', 'Pipeline Item Completed')) AS completed,
               count(*) FILTER (tracking_event IN ('Pipeline Run Errored', 'Pipeline Item Errored'))   AS errored,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND (tracking_event LIKE 'Pipeline Item%'
              OR (tracking_event LIKE 'Pipeline Run%' AND NOT {_RUN_SCOPE} AND NOT {_RECOVERED}))
        GROUP BY ALL ORDER BY day, started DESC
    """,
    # SDK-level operation health (search/add/cognify) by day and version.
    "sdk_exec_outcomes_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version,
               regexp_extract(tracking_event, 'cognee\\.(\\w+) EXECUTION', 1) AS operation,
               count(*) FILTER (tracking_event LIKE '%STARTED')   AS started,
               count(*) FILTER (tracking_event LIKE '%COMPLETED') AS completed,
               count(*) FILTER (tracking_event LIKE '%ERRORED')   AS errored
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event LIKE 'cognee.% EXECUTION%'
        GROUP BY ALL ORDER BY day, operation, version
    """,
    # FastAPI surface: which routes are hit (endpoint is a route template
    # constant like 'POST /v1/search' — no user data), by day.
    "api_endpoint_daily": f"""
        SELECT ingestion_date AS day, {_ENDPOINT} AS endpoint, {_VERSION} AS version,
               count(*) AS events,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event LIKE '%API Endpoint Invoked'
              AND endpoint IS NOT NULL
        GROUP BY ALL ORDER BY day, events DESC
    """,
    # Provider/model settings can contain custom deployment identifiers. Redact
    # before GROUP BY so run and distinct-identity counts cover the whole bucket.
    "provider_stack_daily": f"""
        SELECT ingestion_date AS day, {_EVENT_SCOPE} AS event_scope,
               {_provider_dimension("llm.provider")} AS llm_provider,
               {_provider_dimension("llm.model", max_length=60)} AS llm_model,
               {_LLM_CONFIGURED} AS llm_configured,
               {_STRUCTURED_OUTPUT} AS structured_output,
               {_provider_dimension("embedding.provider")} AS embedding_provider,
               {_provider_dimension("embedding.model", max_length=60)} AS embedding_model,
               {_provider_dimension("graph_extractor")} AS graph_extractor,
               {_provider_dimension("graph.provider")} AS graph_provider,
               {_provider_dimension("vector.provider")} AS vector_provider,
               {_provider_dimension("relational.provider")} AS relational_provider,
               {_VERSION} AS version,
               count(*) FILTER (tracking_event = 'Pipeline Run Started') AS started_runs,
               count(*) FILTER (tracking_event = 'Pipeline Run Completed') AS completed_runs,
               count(*) FILTER (tracking_event = 'Pipeline Run Errored') AS errored_runs,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event LIKE 'Pipeline Run%'
        GROUP BY ALL ORDER BY day, completed_runs DESC
    """,
    # Search-type mix (SearchType enum values only).
    "search_type_daily": f"""
        SELECT ingestion_date AS day, search_type, {_VERSION} AS version,
               count(*) AS events
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event = 'Search API Endpoint Invoked'
              AND search_type IS NOT NULL
        GROUP BY ALL ORDER BY day, events DESC
    """,
    # The memory API's read side (SDK-775): how recall is called, by day, version
    # and surface. ``search_type`` is the SearchType name or ``auto``; ``scope`` the
    # comma-joined source list (``graph``, ``session``, ``trace``, ``code``, ...).
    "recall_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version, {_ORIGIN} AS origin,
               {_closed_value("search_type")} AS search_type,
               {_closed_value("scope", "^[a-z_,]+$")} AS scope,
               {_closed_value("auto_route", "^(true|false)$")} AS auto_route,
               count(*) AS events,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event = 'cognee.recall'
        GROUP BY ALL ORDER BY day, events DESC
    """,
    # The self-improvement loop (SDK-775): improve() calls by day, version and
    # surface, with how many sessions each call bridged (bucketed).
    "improve_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version, {_ORIGIN} AS origin,
               {_closed_value("run_in_background", "^(true|false)$")} AS run_in_background,
               {_SESSION_BUCKET} AS session_count_bucket,
               count(*) AS events,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event = 'cognee.improve'
        GROUP BY ALL ORDER BY day, events DESC
    """,
    # Failures of the SDK operations that have a terminal error event (SDK-775):
    # search and recall, by error class. cognify failures are pipeline events
    # (pipeline_error_types_daily); no cognify EXECUTION event is emitted.
    "sdk_error_types_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version, {_ORIGIN} AS origin,
               tracking_event, {_EXCEPTION_TYPE} AS exception_type,
               {_EXCEPTION_CAUSE} AS exception_cause,
               {_STATUS_CODE} AS status_code,
               count(*) AS errors,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event IN (
            'cognee.search EXECUTION ERRORED', 'cognee.recall ERRORED')
        GROUP BY ALL ORDER BY day, errors DESC
    """,
    # HTTP failures (SDK-775): the API layer's exception event, by route template,
    # status and error class. ``endpoint`` is a route constant, never a URL.
    "api_exceptions_daily": f"""
        SELECT ingestion_date AS day, {_ENDPOINT} AS endpoint, {_VERSION} AS version,
               {_closed_value("status_code", "^[1-5][0-9][0-9]$")} AS status_code,
               {_EXCEPTION_TYPE} AS exception_type,
               count(*) AS events,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event = 'API Exception Raised'
              AND endpoint IS NOT NULL
        GROUP BY ALL ORDER BY day, events DESC
    """,
    # The keyless first run (SDK-775): GLiNER runtime installs started, completed
    # and failed, by platform. Versions and platform names only.
    "gliner_install_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version, tracking_event,
               {_closed_value("os")} AS os,
               {_closed_value("arch")} AS arch,
               {_closed_value("torch_index")} AS torch_index,
               {_closed_value("python_version", "^[0-9]+[.][0-9]+([.][0-9]+)?$")} AS python_version,
               count(*) AS events,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event LIKE 'GLiNER Runtime Install%'
        GROUP BY ALL ORDER BY day, tracking_event
    """,
    # How long pipeline runs take (SDK-775): Started to terminal event per
    # ``pipeline_run_id``, in seconds, as percentiles per day and version. Only
    # runs that ended inside the window are timed; silent runs are counted in
    # pipeline_runs_daily instead.
    "pipeline_run_durations_daily": f"""
        {_RUNS_CTE}
        SELECT day, version,
               count(*) AS runs_timed,
               round(quantile_cont(epoch(ended_at - started_at), 0.5), 1) AS p50_seconds,
               round(quantile_cont(epoch(ended_at - started_at), 0.95), 1) AS p95_seconds,
               round(max(epoch(ended_at - started_at)), 1) AS max_seconds
        FROM runs
        WHERE day IS NOT NULL AND started_at IS NOT NULL AND ended_at IS NOT NULL
              AND NOT recovered AND ended_at >= started_at
        GROUP BY ALL ORDER BY day, version
    """,
    # Which task fails (SDK-775): per-task error events by task name and error
    # class. Task events are not in the allowlist (they are most of the volume:
    # millions of rows a week), so this query filters them on its own and reads
    # at most TASK_WINDOW_DAYS, enough for a week-over-week comparison.
    "task_error_types_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version,
               {_TASK_NAME} AS task_name, {_EXCEPTION_TYPE} AS exception_type,
               {_EXCEPTION_CAUSE} AS exception_cause,
               {_STATUS_CODE} AS status_code,
               count(*) AS errors,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE ingestion_date >= current_date - INTERVAL {TASK_WINDOW_DAYS} DAY
              AND tracking_event LIKE '% Task Errored'
        GROUP BY ALL ORDER BY day, errors DESC
    """,
    # Version lifecycle within the window (adoption/abandonment).
    "version_lifecycle": f"""
        SELECT {_VERSION} AS version,
               (cognee_version LIKE '%-local') AS self_hosted,
               min(ingestion_date) AS first_seen,
               max(ingestion_date) AS last_seen,
               count(*) AS events,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER}
        GROUP BY ALL ORDER BY events DESC
    """,
}

# ---- Output guards -----------------------------------------------------------

HEADER_DENYLIST = re.compile(
    r"(search_query|system_prompt|properties|dataset|user_id|api_key|anonymous"
    r"|persistent|tenant|email|error_text|query)",
    re.IGNORECASE,
)


def _guard(path: Path) -> None:
    """Fail hard if an output file leaks a denylisted column or identifier-shaped cell."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        for column in header:
            if HEADER_DENYLIST.search(column):
                sys.exit(f"PRIVACY GUARD: denylisted column '{column}' in {path.name}")
        for row_number, row in enumerate(reader, start=2):
            for cell in row:
                for pattern in CELL_PATTERNS:
                    if pattern.search(cell):
                        sys.exit(
                            f"PRIVACY GUARD: identifier-shaped value in {path.name}:"
                            f"{row_number} — refusing to publish aggregates"
                        )


def main() -> None:
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        sys.exit("MOTHERDUCK_TOKEN is not set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(f"md:?motherduck_token={token}", read_only=True)

    for name, sql in QUERIES.items():
        out_path = OUT_DIR / f"{name}.csv"
        connection.execute(f"COPY ({sql}) TO '{out_path}' (HEADER, DELIMITER ',')")
        _guard(out_path)
        print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")

    (OUT_DIR / "WINDOW.txt").write_text(
        f"window_days={WINDOW_DAYS}\nnote=aggregates only; identities counted, never exported\n"
    )


if __name__ == "__main__":
    main()
