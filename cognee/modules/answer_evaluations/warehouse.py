"""Only the worker connects to DuckDB/MotherDuck. Stores encrypted payloads."""

import os
from pathlib import Path


class Warehouse:
    def __init__(self, connection):
        self.connection = connection
        connection.execute("CREATE SCHEMA IF NOT EXISTS answer_quality")
        connection.execute("""CREATE TABLE IF NOT EXISTS answer_quality.traces (
            scope_id VARCHAR, trace_id VARCHAR, expires_at DOUBLE, ciphertext VARCHAR,
            PRIMARY KEY(scope_id, trace_id))""")

    @classmethod
    def configured(cls):
        import duckdb

        path = os.environ.get("COGNEE_EVALUATION_WAREHOUSE", "")
        if not path:
            raise ValueError("Set COGNEE_EVALUATION_WAREHOUSE to a local file or md: database")
        if not path.startswith("md:"):
            Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        return cls(duckdb.connect(path))

    def replace(self, scope_id, rows, now):
        con = self.connection
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM answer_quality.traces WHERE expires_at <= ?", [now])
            con.execute("DELETE FROM answer_quality.traces WHERE scope_id = ?", [scope_id])
            for row in rows:
                con.execute(
                    "INSERT INTO answer_quality.traces VALUES (?, ?, ?, ?)",
                    [scope_id, row.id, row.expires_at, row.ciphertext],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    def get(self, scope_id, trace_id, now):
        row = self.connection.execute(
            """SELECT ciphertext FROM answer_quality.traces
            WHERE scope_id = ? AND trace_id = ? AND expires_at > ?""",
            [scope_id, trace_id, now],
        ).fetchone()
        return row[0] if row else None
