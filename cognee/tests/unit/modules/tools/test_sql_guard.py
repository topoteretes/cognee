import pytest

from cognee.modules.tools.errors import SqlGuardError
from cognee.modules.tools.text_to_sql.sql_guard import ensure_limit, strip_sql, validate_select


class TestStripSql:
    def test_removes_markdown_fences(self):
        assert strip_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_removes_comments(self):
        assert strip_sql("SELECT 1 -- trailing comment") == "SELECT 1"
        assert strip_sql("/* leading */ SELECT 1") == "SELECT 1"

    def test_removes_trailing_semicolon(self):
        assert strip_sql("SELECT 1;") == "SELECT 1"

    def test_none_raises(self):
        with pytest.raises(SqlGuardError):
            strip_sql(None)


class TestValidateSelect:
    def test_plain_select_passes(self):
        assert validate_select("SELECT a, b FROM t WHERE a > 1") == "SELECT a, b FROM t WHERE a > 1"

    def test_cte_select_passes(self):
        sql = "WITH x AS (SELECT 1 AS n) SELECT n FROM x"
        assert validate_select(sql) == sql

    def test_empty_rejected(self):
        with pytest.raises(SqlGuardError):
            validate_select("   ")

    def test_non_select_statement_rejected(self):
        for bad in (
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET a = 1",
            "DELETE FROM t",
            "DROP TABLE t",
            "ALTER TABLE t ADD COLUMN a INT",
            "CREATE TABLE t (a INT)",
            "TRUNCATE t",
            "GRANT ALL ON t TO u",
            "PRAGMA table_info('t')",
            "VACUUM",
            "CALL something()",
        ):
            with pytest.raises(SqlGuardError):
                validate_select(bad)

    def test_multi_statement_rejected(self):
        with pytest.raises(SqlGuardError):
            validate_select("SELECT 1; SELECT 2")

    def test_write_hidden_in_cte_rejected(self):
        with pytest.raises(SqlGuardError):
            validate_select("WITH x AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM x")

    def test_write_hidden_behind_comment_rejected(self):
        with pytest.raises(SqlGuardError):
            validate_select("SELECT 1 /* harmless */; DROP TABLE t")

    def test_forbidden_word_inside_string_literal_passes(self):
        sql = "SELECT * FROM logs WHERE message = 'please drop table x'"
        assert validate_select(sql) == sql

    def test_forbidden_word_as_column_substring_passes(self):
        sql = "SELECT updated_at, deleted_flag, created_by FROM t"
        assert validate_select(sql) == sql

    def test_escaped_quote_in_literal_does_not_hide_writes(self):
        with pytest.raises(SqlGuardError):
            validate_select("SELECT 'it''s'; DELETE FROM t")


class TestEnsureLimit:
    def test_appends_limit_when_missing(self):
        assert ensure_limit("SELECT * FROM t", 25) == "SELECT * FROM t LIMIT 25"

    def test_clamps_oversized_limit(self):
        assert ensure_limit("SELECT * FROM t LIMIT 9999", 25) == "SELECT * FROM t LIMIT 25"

    def test_keeps_smaller_limit(self):
        assert ensure_limit("SELECT * FROM t LIMIT 5", 25) == "SELECT * FROM t LIMIT 5"

    def test_case_insensitive(self):
        assert ensure_limit("select * from t limit 500", 25) == "select * from t LIMIT 25"
