"""Unit tests for the Critical + High + Medium/Low security fixes.

Covers:
  * C-2 get_auth_secret fail-hard behavior
  * C-1 graph schema identifier validation
  * H-2 SSRF URL guard
  * H-3 local-file allowlist
  * M-1 SQL column-type allowlist (DDL injection guard)
  * M-2 Cypher / natural-language search secure-by-default gate
  * L-1 Zip Slip guard for the downloaded UI bundle
"""

import io
import os
import tempfile
import zipfile

import pytest

from cognee.modules.ingestion.exceptions import IngestionError


# --------------------------------------------------------------------------- #
# C-2 — auth secrets
# --------------------------------------------------------------------------- #
class TestGetAuthSecret:
    def test_returns_configured_value(self, monkeypatch):
        from cognee.modules.users.authentication.secret_utils import get_auth_secret

        monkeypatch.setenv("SOME_SECRET", "a-real-secret")
        assert get_auth_secret("SOME_SECRET") == "a-real-secret"

    def test_raises_in_production_when_unset(self, monkeypatch):
        from cognee.modules.users.authentication.secret_utils import get_auth_secret

        monkeypatch.setenv("ENV", "prod")
        monkeypatch.delenv("SOME_SECRET", raising=False)
        with pytest.raises(RuntimeError):
            get_auth_secret("SOME_SECRET")

    def test_raises_in_production_on_insecure_default(self, monkeypatch):
        from cognee.modules.users.authentication.secret_utils import get_auth_secret

        monkeypatch.setenv("ENV", "prod")
        monkeypatch.setenv("SOME_SECRET", "super_secret")
        with pytest.raises(RuntimeError):
            get_auth_secret("SOME_SECRET")

    def test_dev_fallback_when_not_production(self, monkeypatch):
        from cognee.modules.users.authentication.secret_utils import get_auth_secret

        monkeypatch.setenv("ENV", "dev")
        monkeypatch.delenv("SOME_SECRET", raising=False)
        assert get_auth_secret("SOME_SECRET")  # non-empty dev default, no raise


# --------------------------------------------------------------------------- #
# C-1 — graph schema validation
# --------------------------------------------------------------------------- #
class TestValidateGraphSchema:
    def _valid_schema(self):
        return {
            "title": "ProgrammingLanguage",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "$defs": {"FieldType": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }

    def test_valid_schema_passes(self):
        from cognee.shared.graph_model_utils import _validate_graph_schema

        _validate_graph_schema(self._valid_schema())

    def test_bad_title_rejected(self):
        from cognee.shared.graph_model_utils import _validate_graph_schema

        schema = self._valid_schema()
        schema["title"] = "x = __import__('os')"
        with pytest.raises(ValueError):
            _validate_graph_schema(schema)

    def test_bad_property_name_rejected(self):
        from cognee.shared.graph_model_utils import _validate_graph_schema

        schema = self._valid_schema()
        schema["properties"] = {"bad-name": {"type": "string"}}
        with pytest.raises(ValueError):
            _validate_graph_schema(schema)

    def test_bad_def_name_rejected(self):
        from cognee.shared.graph_model_utils import _validate_graph_schema

        schema = self._valid_schema()
        schema["$defs"] = {"import os": {"type": "object"}}
        with pytest.raises(ValueError):
            _validate_graph_schema(schema)


# --------------------------------------------------------------------------- #
# H-2 — SSRF URL guard
# --------------------------------------------------------------------------- #
def _fake_getaddrinfo(ip):
    def _inner(host, port, *args, **kwargs):
        return [(2, 1, 6, "", (ip, port or 80))]

    return _inner


class TestAssertUrlAllowed:
    def test_public_url_allowed(self, monkeypatch):
        import cognee.tasks.ingestion.url_safety as url_safety

        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
        monkeypatch.setattr(url_safety.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34"))
        url_safety.assert_url_allowed("http://example.com/page")

    @pytest.mark.parametrize("ip", ["127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.2"])
    def test_internal_ip_blocked(self, monkeypatch, ip):
        import cognee.tasks.ingestion.url_safety as url_safety

        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
        monkeypatch.setattr(url_safety.socket, "getaddrinfo", _fake_getaddrinfo(ip))
        with pytest.raises(IngestionError):
            url_safety.assert_url_allowed("http://internal.example/")

    def test_disabled_when_flag_off(self, monkeypatch):
        import cognee.tasks.ingestion.url_safety as url_safety

        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")
        with pytest.raises(IngestionError):
            url_safety.assert_url_allowed("http://example.com/")

    def test_non_http_scheme_blocked(self, monkeypatch):
        import cognee.tasks.ingestion.url_safety as url_safety

        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
        with pytest.raises(IngestionError):
            url_safety.assert_url_allowed("ftp://example.com/")


# --------------------------------------------------------------------------- #
# H-3 — local-file allowlist
# --------------------------------------------------------------------------- #
class TestAssertLocalPathAllowed:
    def test_path_inside_allowed_root(self, monkeypatch):
        from cognee.tasks.ingestion.url_safety import assert_local_path_allowed

        tmp_dir = tempfile.gettempdir()
        monkeypatch.setenv("LOCAL_FILE_ALLOWED_ROOTS", tmp_dir)
        with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as f:
            path = f.name
        try:
            assert_local_path_allowed(path)  # no raise
        finally:
            os.unlink(path)

    def test_path_outside_allowed_root_blocked(self, monkeypatch):
        from cognee.tasks.ingestion.url_safety import assert_local_path_allowed

        monkeypatch.setenv("LOCAL_FILE_ALLOWED_ROOTS", tempfile.gettempdir())
        with pytest.raises(IngestionError):
            assert_local_path_allowed("/etc/passwd")


# --------------------------------------------------------------------------- #
# M-1 — SQL column-type allowlist (DDL injection guard)
# --------------------------------------------------------------------------- #
class TestValidateSqlType:
    @pytest.mark.parametrize(
        "sql_type",
        ["TEXT", "VARCHAR(255)", "NUMERIC(10, 2)", "integer", "double precision", "INT[]"],
    )
    def test_valid_types_pass(self, sql_type):
        from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
            _validate_sql_type,
        )

        assert _validate_sql_type(sql_type) == sql_type.strip()

    @pytest.mark.parametrize(
        "sql_type",
        [
            "TEXT); DROP TABLE users;--",
            "TEXT DEFAULT (SELECT password FROM users)",
            "'; --",
            "",
            "INT) --",
        ],
    )
    def test_injection_types_rejected(self, sql_type):
        from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
            _validate_sql_type,
        )

        with pytest.raises(ValueError):
            _validate_sql_type(sql_type)


# --------------------------------------------------------------------------- #
# M-2 — Cypher / natural-language search secure-by-default gate
# --------------------------------------------------------------------------- #
class TestCypherQueriesAllowed:
    def _fn(self):
        from cognee.modules.search.methods.get_search_type_retriever_instance import (
            _cypher_queries_allowed,
        )

        return _cypher_queries_allowed

    def test_disabled_by_default_in_production(self, monkeypatch):
        monkeypatch.setenv("ENV", "prod")
        monkeypatch.delenv("ALLOW_CYPHER_QUERY", raising=False)
        assert self._fn()() is False

    def test_enabled_by_default_outside_production(self, monkeypatch):
        monkeypatch.setenv("ENV", "dev")
        monkeypatch.delenv("ALLOW_CYPHER_QUERY", raising=False)
        assert self._fn()() is True

    def test_explicit_true_enables_even_in_production(self, monkeypatch):
        monkeypatch.setenv("ENV", "prod")
        monkeypatch.setenv("ALLOW_CYPHER_QUERY", "true")
        assert self._fn()() is True

    @pytest.mark.parametrize("value", ["false", "no", "0", "off", "maybe", ""])
    def test_non_truthy_values_fail_safe(self, monkeypatch, value):
        monkeypatch.setenv("ENV", "dev")
        monkeypatch.setenv("ALLOW_CYPHER_QUERY", value)
        assert self._fn()() is False


# --------------------------------------------------------------------------- #
# L-1 — Zip Slip guard (downloaded UI bundle extraction)
# --------------------------------------------------------------------------- #
class TestZipSlipGuard:
    def _extract_with_guard(self, names, dest):
        """Replicates the guard applied in cognee/api/v1/ui/ui.py before extractall."""
        import os as _os
        from pathlib import Path

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for n in names:
                zf.writestr(n, "x")
        buf.seek(0)

        extract_dir = Path(dest)
        extract_root = extract_dir.resolve()
        with zipfile.ZipFile(buf, "r") as zip_file:
            for member in zip_file.namelist():
                target = (extract_dir / member).resolve()
                if target != extract_root and extract_root not in target.parents:
                    raise ValueError(f"Unsafe path in downloaded UI archive: {member!r}")
            zip_file.extractall(extract_dir)

    def test_safe_archive_extracts(self):
        with tempfile.TemporaryDirectory() as d:
            self._extract_with_guard(["cognee-1.0/cognee-frontend/index.html"], d)

    @pytest.mark.parametrize("evil", ["../evil.txt", "../../etc/passwd", "a/../../evil"])
    def test_zip_slip_member_rejected(self, evil):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError):
                self._extract_with_guard([evil], d)
