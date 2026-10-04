"""Exercise the telemetry Action's SQL and privacy guard without warehouse access.

Run directly with the Action's DuckDB dependency; no Cognee runtime is needed.
"""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import duckdb
except ImportError:
    duckdb = None


@unittest.skipIf(duckdb is None, "Requires the telemetry Action's DuckDB dependency")
class TelemetryAggregateExtractTest(unittest.TestCase):
    def setUp(self):
        script = (
            Path(__file__).resolve().parents[3] / ".github/scripts/telemetry_aggregate_extract.py"
        )
        spec = importlib.util.spec_from_file_location("telemetry_aggregate_extract", script)
        self.extract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.extract)
        self.connection = duckdb.connect()
        self.addCleanup(self.connection.close)
        self.connection.execute("ATTACH ':memory:' AS analytics")
        self.connection.execute("""
            CREATE TABLE analytics.main.pipeline_events (
                ingestion_date DATE, tracking_event VARCHAR, cognee_version VARCHAR,
                properties JSON, user_id VARCHAR, endpoint VARCHAR, search_type VARCHAR,
                event_timestamp TIMESTAMP, task_name VARCHAR
            )
        """)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.out_dir = Path(directory.name)

    def _insert(self, model="openai/gpt-5-mini", user="deployment-a", **providers):
        properties = {
            "llm": {"provider": "openai", "model": model},
            "graph": {"provider": "kuzu"},
            "vector": {"provider": "lancedb"},
            "relational": {"provider": "sqlite"},
        }
        for provider, value in providers.items():
            properties[provider]["provider"] = value
        self._insert_event("Pipeline Run Completed", "0.5.0-local", properties, user)

    def _provider_rows(self):
        return self._rows("provider_stack_daily")

    def _rows(self, query_name):
        result = self.connection.execute(self.extract.QUERIES[query_name])
        columns = [column[0] for column in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def _insert_event(
        self,
        tracking_event,
        version,
        properties,
        user="deployment-a",
        endpoint=None,
        event_timestamp=None,
        task_name=None,
    ):
        self.connection.execute(
            """INSERT INTO analytics.main.pipeline_events
               (ingestion_date, tracking_event, cognee_version, properties, user_id,
                endpoint, event_timestamp, task_name)
               VALUES (current_date, ?, ?, ?, ?, ?, ?, ?)""",
            [
                tracking_event,
                version,
                json.dumps(properties),
                user,
                endpoint,
                event_timestamp,
                task_name,
            ],
        )

    def test_error_types_are_class_names_or_buckets(self):
        self._insert_event("Pipeline Run Errored", "1.6.0", {"exception_type": "ValueError"})
        self._insert_event("Pipeline Run Errored", "1.6.0", {"exception_type": "ValueError"}, "b")
        self._insert_event("Pipeline Run Errored", "1.6.0", {})
        self._insert_event(
            "Pipeline Run Errored", "1.6.0", {"exception_type": "person@example.com"}
        )
        self._insert_event("Pipeline Run Completed", "1.6.0", {"exception_type": "ValueError"})
        rows = {row["exception_type"]: row for row in self._rows("pipeline_error_types_daily")}
        self.assertEqual(set(rows), {"ValueError", "unknown", "redacted"})
        self.assertEqual(rows["ValueError"]["errors"], 2)
        self.assertEqual(rows["ValueError"]["distinct_identities"], 2)
        self.assertEqual(rows["unknown"]["errors"], 1)
        self.assertEqual(rows["redacted"]["errors"], 1)

    def test_embedding_and_extractor_dimensions_are_redacted_like_the_llm_ones(self):
        self._insert_event(
            "Pipeline Run Completed",
            "1.6.0",
            {
                "llm": {"provider": "openai", "model": "gpt"},
                "embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
                "graph_extractor": "gliner_demo",
                "graph": {"provider": "kuzu"},
                "vector": {"provider": "lancedb"},
                "relational": {"provider": "sqlite"},
            },
        )
        self._insert_event(
            "Pipeline Run Completed",
            "1.6.0",
            {
                "llm": {"provider": "openai", "model": "gpt"},
                "embedding": {"provider": "custom", "model": "custom/person@example.com"},
                "graph_extractor": "llm",
                "graph": {"provider": "kuzu"},
                "vector": {"provider": "lancedb"},
                "relational": {"provider": "sqlite"},
            },
            "b",
        )
        rows = {row["graph_extractor"]: row for row in self._provider_rows()}
        self.assertEqual(rows["gliner_demo"]["embedding_provider"], "fastembed")
        self.assertEqual(rows["gliner_demo"]["embedding_model"], "baai/bge-small-en-v1.5")
        self.assertEqual(rows["llm"]["embedding_model"], "redacted")

    def test_runs_are_classified_once_from_all_their_events(self):
        """Events fire per data item and recovery closes a run with one event; runs
        are counted by pipeline_run_id so a killed multi-item run balances."""

        def run(run_id, *events):
            for event in events:
                self._insert_event(f"Pipeline Run {event}", "1.6.0", {"pipeline_run_id": run_id})

        run("run-completed", "Started", "Started", "Completed", "Completed")
        run("run-killed", "Started", "Started", "Started", "Errored")  # recovery: one event
        run("run-mixed", "Started", "Started", "Completed", "Errored")
        run("run-silent", "Started")
        run("run-no-start", "Completed")  # its Started never arrived: not counted
        self._insert_event("Pipeline Run Started", "1.5.4", {})  # legacy, no run id

        (row,) = self._rows("pipeline_runs_daily")
        self.assertEqual(row["version"], "1.6.0")
        self.assertEqual(row["runs_started"], 4)
        self.assertEqual(row["runs_completed"], 1)
        self.assertEqual(row["runs_errored"], 2)
        self.assertEqual(row["runs_silent"], 1)

    def test_error_types_count_items_and_runs_separately(self):
        for run_id in ("run-a", "run-a", "run-b"):
            self._insert_event(
                "Pipeline Run Errored",
                "1.6.0",
                {"exception_type": "ValueError", "pipeline_run_id": run_id},
            )
        (row,) = self._rows("pipeline_error_types_daily")
        self.assertEqual(row["errors"], 3)
        self.assertEqual(row["runs"], 2)

    def test_redacts_identifiers_in_provider_dimensions(self):
        for value in (
            "custom/person@example.com",
            "custom/12345678-1234-1234-1234-123456789abc",
            "custom/ak_0123456789abcdef",
        ):
            for dimension in ("llm_model", "llm", "graph", "vector", "relational"):
                with self.subTest(value=value, dimension=dimension):
                    self.connection.execute("DELETE FROM analytics.main.pipeline_events")
                    if dimension == "llm_model":
                        self._insert(model=value)
                        column = dimension
                    else:
                        self._insert(**{dimension: value})
                        column = f"{dimension}_provider"
                    row = self._provider_rows()[0]
                    self.assertEqual(row[column], "redacted")
                    self.assertEqual(row["completed_runs"], 1)

    def test_redacts_models_when_truncation_changes_identifier_boundaries(self):
        for model in (
            "prefix-" * 8 + "12345678-1234-1234-1234-123456789ABC",
            "prefix-" + "x" * 16 + "/12345678-1234-1234-1234-123456789abcde",
        ):
            with self.subTest(model=model):
                self.connection.execute("DELETE FROM analytics.main.pipeline_events")
                self._insert(model=model)
                self.assertEqual(self._provider_rows()[0]["llm_model"], "redacted")

    def test_groups_redacted_models_before_counting_distinct_identities(self):
        first = "custom/12345678-1234-1234-1234-123456789abc"
        second = "custom/87654321-4321-4321-4321-cba987654321"
        self._insert(model=first, user="deployment-a")
        self._insert(model=second, user="deployment-a")
        self._insert(model=second, user="deployment-b")
        rows = self._provider_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["llm_model"], "redacted")
        self.assertEqual(rows[0]["completed_runs"], 3)
        self.assertEqual(rows[0]["distinct_identities"], 2)

    def test_preserves_normal_model_normalization_and_missing_values(self):
        for model in ("OpenAI/GPT-4o-2024-08-06", "model-" * 15, None):
            with self.subTest(model=model):
                self.connection.execute("DELETE FROM analytics.main.pipeline_events")
                self._insert(model=model)
                row = self._provider_rows()[0]
                self.assertEqual(row["llm_model"], model.lower()[:60] if model else None)
                self.assertEqual(row["llm_provider"], "openai")
                self.assertEqual(row["version"], "0.5.0")

    def test_extracts_all_csvs_with_identifier_bearing_model(self):
        self._insert(model="custom/person@example.com")
        with (
            patch.dict("os.environ", {"MOTHERDUCK_TOKEN": "test-token"}),
            patch.object(self.extract, "OUT_DIR", self.out_dir),
            patch.object(self.extract.duckdb, "connect", return_value=self.connection),
        ):
            self.extract.main()
        self.assertEqual(
            {path.stem for path in self.out_dir.glob("*.csv")}, set(self.extract.QUERIES)
        )
        for path in self.out_dir.glob("*.csv"):
            self.extract._guard(path)
            self.assertNotIn("person@example.com", path.read_text())
        self.assertTrue((self.out_dir / "WINDOW.txt").is_file())

    def test_guard_still_rejects_identifiers_without_echoing_them(self):
        for value in (
            "person@example.com",
            "12345678-1234-1234-1234-123456789abc",
            "ak_0123456789abcdef",
        ):
            with self.subTest(value=value):
                path = self.out_dir / "unsafe.csv"
                with path.open("w", newline="") as handle:
                    csv.writer(handle).writerows([["llm_model"], [value]])
                with self.assertRaisesRegex(SystemExit, "PRIVACY GUARD") as error:
                    self.extract._guard(path)
                self.assertNotIn(value, str(error.exception))

    def test_guard_still_rejects_identity_columns(self):
        path = self.out_dir / "unsafe.csv"
        path.write_text("user_id\n1\n")
        with self.assertRaisesRegex(SystemExit, "denylisted column"):
            self.extract._guard(path)

    # ---- SDK-775: the memory API, errors by class, durations, installs ----------

    def test_recall_rows_carry_closed_values_only(self):
        self._insert_event(
            "cognee.recall",
            "1.6.3",
            {"search_type": "auto", "scope": "session,graph", "auto_route": True},
        )
        self._insert_event(
            "cognee.recall",
            "1.6.3",
            {"search_type": "GRAPH_COMPLETION", "scope": "graph", "auto_route": False},
            "b",
        )
        self._insert_event(
            "cognee.recall",
            "1.6.3",
            {"search_type": "person@example.com", "scope": "graph; drop", "auto_route": "maybe"},
            "c",
        )
        rows = {row["search_type"]: row for row in self._rows("recall_daily")}
        self.assertEqual(set(rows), {"auto", "GRAPH_COMPLETION", "redacted"})
        self.assertEqual(rows["auto"]["scope"], "session,graph")
        self.assertEqual(rows["auto"]["auto_route"], "true")
        self.assertEqual(rows["redacted"]["scope"], "redacted")
        self.assertEqual(rows["redacted"]["auto_route"], "redacted")

    def test_improve_rows_bucket_session_counts(self):
        for count, user in ((0, "a"), (1, "b"), (4, "c"), (9, "d")):
            self._insert_event(
                "cognee.improve",
                "1.6.3",
                {"session_count": count, "run_in_background": False},
                user,
            )
        self._insert_event("cognee.improve", "1.6.3", {}, "e")
        buckets = {row["session_count_bucket"] for row in self._rows("improve_daily")}
        self.assertEqual(buckets, {"0", "1", "2-5", "6+", "unknown"})

    def test_sdk_error_types_cover_search_and_recall_only(self):
        self._insert_event(
            "cognee.search EXECUTION ERRORED", "1.6.3", {"exception_type": "PermissionDeniedError"}
        )
        self._insert_event(
            "cognee.recall ERRORED", "1.6.3", {"exception_type": "CancelledError"}, "b"
        )
        self._insert_event("cognee.search EXECUTION COMPLETED", "1.6.3", {}, "d")
        rows = {
            (row["tracking_event"], row["exception_type"])
            for row in self._rows("sdk_error_types_daily")
        }
        self.assertEqual(
            rows,
            {
                ("cognee.search EXECUTION ERRORED", "PermissionDeniedError"),
                ("cognee.recall ERRORED", "CancelledError"),
            },
        )

    def test_api_exceptions_group_by_route_status_and_class(self):
        for status, user in ((404, "a"), (404, "b"), (500, "c")):
            self._insert_event(
                "API Exception Raised",
                "1.6.3",
                {"status_code": status, "exception_type": "DatasetNotFoundError"},
                user,
                endpoint="GET /api/v1/datasets/{dataset_id}",
            )
        self._insert_event(
            "API Exception Raised",
            "1.6.3",
            {"status_code": "12345", "exception_type": "x y"},
            "d",
            endpoint="POST /api/v1/recall",
        )
        rows = {
            (row["endpoint"], row["status_code"], row["exception_type"]): row
            for row in self._rows("api_exceptions_daily")
        }
        self.assertEqual(
            rows[("GET /api/v1/datasets/{dataset_id}", "404", "DatasetNotFoundError")]["events"], 2
        )
        self.assertEqual(rows[("POST /api/v1/recall", "redacted", "redacted")]["events"], 1)

    def test_gliner_install_rows_keep_platform_names_only(self):
        props = {
            "os": "Darwin",
            "arch": "arm64",
            "torch_index": "pytorch-cpu",
            "python_version": "3.12.4",
        }
        self._insert_event("GLiNER Runtime Install Started", "1.6.3", props)
        self._insert_event("GLiNER Runtime Install Completed", "1.6.3", props)
        self._insert_event(
            "GLiNER Runtime Install Failed",
            "1.6.3",
            {**props, "torch_index": "https://mirror.example.com/x"},
            "b",
        )
        rows = {row["tracking_event"]: row for row in self._rows("gliner_install_daily")}
        self.assertEqual(rows["GLiNER Runtime Install Started"]["os"], "Darwin")
        self.assertEqual(rows["GLiNER Runtime Install Completed"]["python_version"], "3.12.4")
        self.assertEqual(rows["GLiNER Runtime Install Failed"]["torch_index"], "redacted")

    def test_run_durations_are_percentiles_per_run(self):
        from datetime import datetime, timedelta, timezone

        start = datetime(2026, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
        for run_id, seconds in (("r1", 10), ("r2", 30), ("r3", 110)):
            self._insert_event(
                "Pipeline Run Started", "1.6.3", {"pipeline_run_id": run_id}, event_timestamp=start
            )
            self._insert_event(
                "Pipeline Run Started",
                "1.6.3",
                {"pipeline_run_id": run_id},
                event_timestamp=start + timedelta(seconds=1),
            )
            terminal = "Pipeline Run Errored" if run_id == "r3" else "Pipeline Run Completed"
            self._insert_event(
                terminal,
                "1.6.3",
                {"pipeline_run_id": run_id},
                event_timestamp=start + timedelta(seconds=seconds),
            )
        self._insert_event(
            "Pipeline Run Started", "1.6.3", {"pipeline_run_id": "silent"}, event_timestamp=start
        )
        self._insert_event("Pipeline Run Started", "1.5.4", {}, event_timestamp=start)  # no run id
        (row,) = self._rows("pipeline_run_durations_daily")
        self.assertEqual(row["runs_timed"], 3)
        self.assertEqual(row["p50_seconds"], 30.0)
        self.assertEqual(row["max_seconds"], 110.0)

    def test_task_errors_group_by_identifier_task_names(self):
        self._insert_event(
            "Coroutine Task Errored",
            "1.6.3",
            {"exception_type": "ValueError"},
            task_name="extract_graph_from_data",
        )
        self._insert_event(
            "Coroutine Task Errored",
            "1.6.3",
            {"exception_type": "ValueError"},
            "b",
            task_name="extract_graph_from_data",
        )
        self._insert_event(
            "Async Generator Task Errored", "1.6.3", {}, "c", task_name="person@example.com"
        )
        self._insert_event(
            "Coroutine Task Started", "1.6.3", {}, task_name="extract_graph_from_data"
        )
        rows = {
            (row["task_name"], row["exception_type"]): row
            for row in self._rows("task_error_types_daily")
        }
        self.assertEqual(rows[("extract_graph_from_data", "ValueError")]["errors"], 2)
        self.assertEqual(rows[("redacted", "unknown")]["errors"], 1)

    def test_raw_ids_in_endpoints_are_folded_into_a_placeholder(self):
        """Older builds send the raw path; the id is folded before grouping and
        before the guard sees the file."""
        for user in ("a", "a", "b"):
            self._insert_event(
                "Recall API Endpoint Invoked",
                "1.5.4",
                {},
                user,
                endpoint="POST /api/v1/recall/12345678-1234-1234-1234-123456789abc",
            )
        self._insert_event(
            "API Exception Raised",
            "1.5.4",
            {"status_code": 404, "exception_type": "DatasetNotFoundError"},
            endpoint="GET /api/v1/datasets/ABCDEF01-1234-1234-1234-123456789ABC/graph",
        )
        (row,) = self._rows("api_endpoint_daily")
        self.assertEqual(row["endpoint"], "POST /api/v1/recall/{id}")
        self.assertEqual(row["events"], 3)
        (row,) = self._rows("api_exceptions_daily")
        self.assertEqual(row["endpoint"], "GET /api/v1/datasets/{id}/graph")

    def test_path_like_models_are_redacted_and_rejected_by_the_guard(self):
        for model in ("/Users/alice/models/x.gguf", "C:\\Users\\alice\\x.gguf", "/home/alice/m"):
            with self.subTest(model=model):
                self.connection.execute("DELETE FROM analytics.main.pipeline_events")
                self._insert(model=model)
                self.assertEqual(self._provider_rows()[0]["llm_model"], "redacted")
                path = self.out_dir / "unsafe.csv"
                with path.open("w", newline="") as handle:
                    csv.writer(handle).writerows([["llm_model"], [model]])
                with self.assertRaisesRegex(SystemExit, "PRIVACY GUARD"):
                    self.extract._guard(path)
        # route templates stay: lower-case segments are not account directories
        path = self.out_dir / "routes.csv"
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerows(
                [["endpoint"], ["GET /api/v1/users/me"], ["GET /users/me"], ["POST /v1/add"]]
            )
        self.extract._guard(path)
        # ...while the SQL redaction, which lower-cases, still catches a macOS home dir
        self.connection.execute("DELETE FROM analytics.main.pipeline_events")
        self._insert(model="/USERS/alice/x.gguf")
        self.assertEqual(self._provider_rows()[0]["llm_model"], "redacted")


if __name__ == "__main__":
    unittest.main()
