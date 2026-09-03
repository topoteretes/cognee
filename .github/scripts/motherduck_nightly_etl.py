"""Load nightly performance reports from S3 into MotherDuck (SDK-37).

Every performance job already uploads its full report JSON to S3 before the
Slack message is built:

    s3://<bucket>/performance_results/{backend}/{label}/{mode}_{TS}.json

That JSON carries far more than Slack shows — every percentile
(min/max/mean/p50/p75/p90/p95/p99) for every metric, plus per-run detail in
`raw_runs`. This script points MotherDuck at those objects and shapes them into
a small set of views the dashboards read.

Design notes:

* The read is driven off the S3 object list, not off the workflow's job
  outputs. That keeps every percentile (Slack carries only p50/p90/p99 for four
  metrics) and means a first run backfills the whole bucket rather than
  starting from empty.
* `raw_perf_reports` stores each report as a whole JSON value instead of an
  inferred schema. The backends emit genuinely different metric sets (cloud has
  tenant_* and, when it does not create a tenant, no prune_time_s), so a typed
  table would drift and break. All shaping happens in the views.
* The INSERT is an anti-join on the S3 key, so the script is idempotent and the
  backfill and the nightly increment are the same statement.

Env:
  motherduck_token       — MotherDuck token (duckdb reads this exact name)
  MD_TARGET              — target catalog.schema (default: ci_analytics.nightly)
  AWS_ACCESS_KEY_ID      — S3 read credentials for the reports bucket
  AWS_SECRET_ACCESS_KEY
  AWS_SESSION_TOKEN      — optional, only for temporary credentials
  AWS_DEFAULT_REGION     — default: eu-west-1
  PERF_BUCKET            — default: github-runner-cognee-tests
  PERF_PREFIX            — default: performance_results
  PERF_REFRESH_VIEWS     — 'false' skips the CREATE OR REPLACE VIEW pass and
                           only loads new rows. The nightly sets this on the
                           weekly main run so view definitions always come from
                           the daily dev run: two refs running this script would
                           otherwise make them last-writer-wins.
"""

import os
import sys

import duckdb

MD_TARGET = os.environ.get("MD_TARGET", "ci_analytics.nightly")
CATALOG, SCHEMA = MD_TARGET.split(".", 1)
BUCKET = os.environ.get("PERF_BUCKET", "github-runner-cognee-tests")
PREFIX = os.environ.get("PERF_PREFIX", "performance_results").strip("/")
REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
ROOT = f"s3://{BUCKET}/{PREFIX}"

TGT = f'"{CATALOG}"."{SCHEMA}"'

# Two globs rather than '**': the bucket holds both the current
# {backend}/{label}/ layout and the pre-2026-06-04 {label}/ layout that predates
# 9301645b7 ("add postgres to nightly CI runs").
GLOBS = [f"{ROOT}/*/*.json", f"{ROOT}/*/*/*.json"]

VIEWS = {
    # One row per report: the run header.
    "v_perf_runs": """
        WITH parsed AS (
            SELECT s3_key, uploaded_at, report,
                   regexp_extract(s3_key,
                       'performance_results/(?:([^/]+)/)?([^/]+)/([^/]+)\\.json$',
                       ['backend_raw', 'label', 'stem']) AS p
            FROM {t}.raw_perf_reports
        ),
        split AS (
            SELECT *,
                   -- mode can itself contain '_' ("mock_llm"), so anchor on the
                   -- trailing timestamp instead of splitting on the first '_'.
                   regexp_extract(p.stem,
                       '^(.*)_(\\d{{4}}-\\d{{2}}-\\d{{2}}_\\d{{2}}-\\d{{2}}-\\d{{2}}Z)$',
                       ['mode', 'ts']) AS s,
                   coalesce(nullif(p.backend_raw, ''), 'file_based') AS backend,
                   -- Stamped by the perf workflows' "Stamp run provenance" step.
                   -- Rows written before that step existed carry none of it. They
                   -- are defaulted to 'main': the cron only ever fired on the
                   -- default branch, so all but a handful of hand-dispatched
                   -- validation runs really are main, and defaulting this way
                   -- keeps main's baseline continuous.
                   coalesce(nullif(report ->> '$.branch', ''), 'main') AS branch,
                   p.backend_raw = '' AS legacy_path
            FROM parsed
        )
        SELECT
            s3_key,
            -- run_ts is deliberately TIMESTAMP and there is no DATE column:
            -- Superset/Preset cannot serialize a DATE as a pivot key
            -- ("keys must be str, int, float, bool or None, not datetime.date").
            -- Callers that want a day should cast: run_ts::DATE.
            strptime(s.ts, '%Y-%m-%d_%H-%M-%SZ')            AS run_ts,
            backend,
            branch,
            legacy_path,
            CASE WHEN backend LIKE 'rust\\_%' ESCAPE '\\' THEN 'rust' ELSE 'python' END AS sdk,
            regexp_replace(backend, '^rust_', '')           AS store,
            p.label                                         AS label,
            s.mode                                          AS mode,
            concat_ws('/',
                CASE WHEN backend LIKE 'rust\\_%' ESCAPE '\\' THEN 'rust' ELSE 'python' END,
                regexp_replace(backend, '^rust_', ''), p.label, s.mode) AS suite,
            -- `suite` is deliberately UNCHANGED so no existing dashboard chart
            -- breaks. `series` is the branch-qualified key; it is what
            -- v_perf_regression partitions on and what a time-series chart
            -- should group by once its owner migrates. Opt-in, not forced.
            concat_ws('/', branch,
                CASE WHEN backend LIKE 'rust\\_%' ESCAPE '\\' THEN 'rust' ELSE 'python' END,
                regexp_replace(backend, '^rust_', ''), p.label, s.mode) AS series,
            (report ->> '$.num_runs')::INT                   AS num_runs,
            (report ->> '$.succeeded')::INT                  AS succeeded,
            (report ->> '$.failed')::INT                     AS failed,
            (report ->> '$.succeeded')::INT = (report ->> '$.num_runs')::INT AS all_passed,
            report ->> '$.git_sha'                           AS git_sha,
            report ->> '$.run_id'                            AS run_id,
            report ->> '$.run_attempt'                       AS run_attempt,
            report ->> '$.event'                             AS event,
            report ->> '$.config.llm_model'                  AS llm_model,
            report ->> '$.config.embedding_model'            AS embedding_model,
            report ->> '$.config.embedding_dimensions'       AS embedding_dimensions,
            (report ->> '$.config.mock_llm')::BOOLEAN        AS mock_llm,
            report ->> '$.config.tenant_url'                 AS tenant_url,
            uploaded_at
        FROM split
    """,
    # Long format — one row per run x metric x stat. Charts filter to a single
    # metric + stat and group by suite; this survives per-backend metric drift.
    "v_perf_metrics": """
        WITH per_metric AS (
            SELECT r.s3_key, r.run_ts, r.branch, r.series, r.suite, r.sdk, r.store,
                   r.label, r.mode, r.git_sha, r.all_passed,
                   m.metric                            AS metric,
                   raw.report -> '$.stats' -> m.metric AS mstats
            FROM {t}.v_perf_runs r
            JOIN {t}.raw_perf_reports raw USING (s3_key),
                 UNNEST(json_keys(raw.report, '$.stats')) AS m(metric)
        )
        SELECT s3_key, run_ts, branch, series, suite, sdk, store, label, mode, git_sha,
               all_passed, metric, st.stat AS stat, (mstats ->> st.stat)::DOUBLE AS value_s
        FROM per_metric, UNNEST(json_keys(mstats)) AS st(stat)
    """,
    # Latest run vs the median of the previous seven, per series/metric/stat.
    "v_perf_regression": """
        WITH ranked AS (
            SELECT *, row_number() OVER (
                       -- series, not suite: without the branch in the key a
                       -- weekly main run becomes rn=1 for a suite whose baseline
                       -- is seven dev runs, silently.
                       PARTITION BY series, metric, stat ORDER BY run_ts DESC) AS rn
            FROM {t}.v_perf_metrics
            WHERE stat IN ('p50', 'p90', 'p99')
              -- A failed run's placeholder timings must never become a baseline.
              AND all_passed
        )
        -- `suite` is kept (functionally determined by series) so existing
        -- queries still resolve; they now get one row per branch and should
        -- add `WHERE branch = 'dev'`.
        SELECT branch, series, suite, metric, stat,
               max(run_ts) FILTER (WHERE rn = 1)                   AS latest_ts,
               max(value_s) FILTER (WHERE rn = 1)                  AS latest_s,
               median(value_s) FILTER (WHERE rn BETWEEN 2 AND 8)   AS baseline_s,
               -- "the previous 7 runs" is 7 days on the daily dev series but up
               -- to 7 WEEKS on the weekly main series. Read the window before
               -- trusting pct_change.
               min(run_ts)  FILTER (WHERE rn BETWEEN 2 AND 8)      AS baseline_from_ts,
               round(100.0 * (max(value_s) FILTER (WHERE rn = 1)
                     - median(value_s) FILTER (WHERE rn BETWEEN 2 AND 8))
                     / nullif(median(value_s) FILTER (WHERE rn BETWEEN 2 AND 8), 0), 1)
                                                                   AS pct_change
        FROM ranked WHERE rn <= 8
        GROUP BY branch, series, suite, metric, stat
    """,
}


def main() -> int:
    con = duckdb.connect("md:")

    # The S3 read runs on MotherDuck's cloud engine, not on the runner, so the
    # credential has to be registered server-side -- a session-local DuckDB
    # secret is not visible to it. Use a read-only IAM key scoped to this
    # bucket: it is stored (encrypted) in MotherDuck until the next run
    # replaces it.
    params = [
        "TYPE s3",
        f"KEY_ID '{os.environ['AWS_ACCESS_KEY_ID']}'",
        f"SECRET '{os.environ['AWS_SECRET_ACCESS_KEY']}'",
        f"REGION '{REGION}'",
        f"SCOPE 's3://{BUCKET}'",
    ]
    if os.environ.get("AWS_SESSION_TOKEN"):
        params.insert(3, f"SESSION_TOKEN '{os.environ['AWS_SESSION_TOKEN']}'")
    try:
        con.execute(f"CREATE OR REPLACE SECRET cognee_ci_s3 IN MOTHERDUCK ({', '.join(params)});")
    except Exception as exc:  # never echo the statement -- it holds the key
        raise RuntimeError(f"failed to register S3 secret: {type(exc).__name__}") from None
    print(f"registered S3 secret for s3://{BUCKET} ({REGION})")

    con.execute(f'CREATE DATABASE IF NOT EXISTS "{CATALOG}";')
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{CATALOG}"."{SCHEMA}";')
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TGT}.raw_perf_reports (
            s3_key      VARCHAR,
            uploaded_at TIMESTAMP,
            size_bytes  BIGINT,
            report      JSON
        );
        """
    )

    before = con.execute(f"SELECT count(*) FROM {TGT}.raw_perf_reports").fetchone()[0]
    globs = ", ".join(f"'{g}'" for g in GLOBS)
    con.execute(
        f"""
        INSERT INTO {TGT}.raw_perf_reports
        SELECT filename, last_modified, size, content::JSON
        FROM read_text([{globs}])
        WHERE filename NOT IN (SELECT s3_key FROM {TGT}.raw_perf_reports);
        """
    )
    after = con.execute(f"SELECT count(*) FROM {TGT}.raw_perf_reports").fetchone()[0]
    print(f"raw_perf_reports: {after} reports (+{after - before} new)")

    if os.environ.get("PERF_REFRESH_VIEWS", "true").lower() == "false":
        print("PERF_REFRESH_VIEWS=false — rows loaded, views left as-is.")
        return 0

    failures = 0
    for name, sql in VIEWS.items():
        try:
            con.execute(f"CREATE OR REPLACE VIEW {TGT}.{name} AS {sql.format(t=TGT)};")
            print(f"view {name}: created")
        except Exception as exc:
            failures += 1
            print(f"WARN view {name} failed ({type(exc).__name__}): {str(exc).splitlines()[0]}")

    if not failures:
        runs, suites, bad = con.execute(
            f"""SELECT count(*), count(DISTINCT suite),
                       sum(CASE WHEN NOT all_passed THEN 1 ELSE 0 END)
                FROM {TGT}.v_perf_runs"""
        ).fetchone()
        print(f"{MD_TARGET}: {runs} runs across {suites} suites, {bad} with failures")

    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
