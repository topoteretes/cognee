"""Unit tests for the Critical + High security fixes.

Covers:
  * C-2 get_auth_secret fail-hard behavior
  * C-1 graph schema identifier validation
  * H-2 SSRF URL guard
  * H-3 local-file allowlist
"""

import os
import tempfile

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
