"""Tests for DLT internal-table detection in schema extraction.

DLT's own artifacts (bookkeeping tables, staging-schema copies) must be
excluded from the ingested schema by exact prefix/suffix convention — never
by substring, which silently dropped user tables like "staging_orders".
"""

import pytest

from cognee.tasks.ingestion.ingest_dlt_source import _is_dlt_internal_table


class TestIsDltInternalTable:
    @pytest.mark.parametrize(
        "qualified_name",
        [
            "_dlt_loads",  # SQLite bookkeeping (bare table names)
            "_dlt_version",
            "_dlt_pipeline_state",
            "mydataset._dlt_loads",  # Postgres bookkeeping (schema-qualified)
            "mydataset_staging.orders",  # Postgres staging-schema copy
            "mydataset_staging.customers",
        ],
    )
    def test_dlt_artifacts_are_internal(self, qualified_name):
        assert _is_dlt_internal_table(qualified_name) is True

    @pytest.mark.parametrize(
        "qualified_name",
        [
            "orders",
            "mydataset.orders",
            # Regressions: user tables that the old substring filter dropped.
            "staging_orders",
            "mydataset.staging_orders",
            "orders_staging_area",
            "my_dlt_exports",
            "mydataset.my_dlt_exports",
            # "staging" in the dataset/schema name but not DLT's exact suffix.
            "staging_env.orders",
        ],
    )
    def test_user_tables_are_kept(self, qualified_name):
        assert _is_dlt_internal_table(qualified_name) is False
