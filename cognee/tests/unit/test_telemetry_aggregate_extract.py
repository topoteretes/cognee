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
                properties JSON, user_id VARCHAR, endpoint VARCHAR, search_type VARCHAR
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
        self.connection.execute(
            """INSERT INTO analytics.main.pipeline_events VALUES
               (current_date, 'Pipeline Run Completed', '0.5.0-local', ?, ?, NULL, NULL)""",
            [json.dumps(properties), user],
        )

    def _provider_rows(self):
        result = self.connection.execute(self.extract.QUERIES["provider_stack_daily"])
        columns = [column[0] for column in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

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


if __name__ == "__main__":
    unittest.main()
