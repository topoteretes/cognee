"""Extract ANONYMIZED AGGREGATES from the MotherDuck telemetry warehouse.

This script is the privacy boundary of the daily telemetry-insights Action:
the analysis model (Claude Code) never receives warehouse credentials and
never sees a raw event — only the CSV aggregates this script emits.

Hard rules enforced here:
- Only the queries below run; every SELECT lists explicit output columns.
- Free-text / PII-bearing fields are NEVER selected: search_query,
  system_prompt, dataset names, raw properties, tenant ids, endpoints'
  query strings, error text.
- Identity columns (user_id, api_key_hash, anonymous_id, persistent_id)
  are used ONLY inside COUNT(DISTINCT ...); their values are never emitted.
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
OUT_DIR = Path(os.getenv("TELEMETRY_OUT_DIR", "telemetry_aggregates"))

# Events worth analyzing; everything else (internal task/coroutine spam) is skipped.
EVENT_ALLOWLIST = (
    "cognee.search EXECUTION STARTED",
    "cognee.search EXECUTION COMPLETED",
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
    "Pipeline Run Started",
    "Pipeline Run Completed",
    "Pipeline Run Errored",
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
# Surface the event came from: 'sdk' (default), 'cloud', 'cli', ... Safe enum.
_ORIGIN = "coalesce(json_extract_string(properties, '$.telemetry_origin'), 'unknown')"


def _san(expr: str) -> str:
    """Scrub identifier-shaped substrings from a user-configurable string column.

    Provider/model/endpoint/version values come from customer config and can
    embed UUIDs (deployment names), long hex ids, or even email addresses —
    the first scheduled run tripped the privacy guard on exactly this (an
    identifier inside an llm_model string). Replace them with typed tokens so
    the aggregate keys stay useful without carrying identifiers.
    """
    uuid = "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    email = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"
    long_hex = "[0-9a-fA-F]{16,}"
    return (
        f"regexp_replace(regexp_replace(regexp_replace({expr}, "
        f"'{uuid}', '<uuid>', 'g'), "
        f"'{email}', '<email>', 'g'), "
        f"'{long_hex}', '<hex>', 'g')"
    )


# Normalized version: strip the -local suffix so builds compare cleanly.
_VERSION = _san("coalesce(regexp_replace(cognee_version, '-local$', ''), 'unknown')")

_EVENTS_SQL = "(" + ",".join(f"'{e}'" for e in EVENT_ALLOWLIST) + ")"
_BASE_FILTER = (
    f"ingestion_date >= current_date - INTERVAL {WINDOW_DAYS} DAY "
    f"AND tracking_event IN {_EVENTS_SQL}"
)

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
    # Graph-build pipeline health by day and version.
    "pipeline_outcomes_daily": f"""
        SELECT ingestion_date AS day, {_VERSION} AS version,
               count(*) FILTER (tracking_event = 'Pipeline Run Started')   AS started,
               count(*) FILTER (tracking_event = 'Pipeline Run Completed') AS completed,
               count(*) FILTER (tracking_event = 'Pipeline Run Errored')   AS errored,
               count(DISTINCT {_IDENT}) FILTER (tracking_event = 'Pipeline Run Errored')
                   AS identities_with_errors
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event LIKE 'Pipeline Run%'
        GROUP BY ALL ORDER BY day, version
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
        SELECT ingestion_date AS day, {_san("endpoint")} AS endpoint, {_VERSION} AS version,
               count(*) AS events,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event LIKE '%API Endpoint Invoked'
              AND endpoint IS NOT NULL
        GROUP BY ALL ORDER BY day, events DESC
    """,
    # Provider stack correlation (from completed pipeline runs).
    "provider_stack_daily": f"""
        SELECT ingestion_date AS day,
               {_san("json_extract_string(properties, '$.llm.provider')")}   AS llm_provider,
               {_san("left(lower(json_extract_string(properties, '$.llm.model')), 60)")} AS llm_model,
               {_san("json_extract_string(properties, '$.graph.provider')")} AS graph_provider,
               {_san("json_extract_string(properties, '$.vector.provider')")} AS vector_provider,
               {_san("json_extract_string(properties, '$.relational.provider')")} AS relational_provider,
               {_VERSION} AS version,
               count(*) AS completed_runs,
               count(DISTINCT {_IDENT}) AS distinct_identities
        FROM analytics.main.pipeline_events
        WHERE {_BASE_FILTER} AND tracking_event = 'Pipeline Run Completed'
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
CELL_PATTERNS = (
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),  # email
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE),  # uuid
    re.compile(r"\bak_[0-9a-f]{16,}\b", re.IGNORECASE),  # key hash
    re.compile(r"\b[0-9a-fA-F]{16,}\b"),  # bare long hex/opaque id
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
