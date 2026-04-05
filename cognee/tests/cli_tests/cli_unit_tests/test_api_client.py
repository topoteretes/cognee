"""Unit tests for cognee.cli.api_client."""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from cognee.cli.api_client import CogneeApiClient


class TestCogneeApiClientInit:
    """Test client construction and URL helpers."""

    def test_base_url_trailing_slash_stripped(self):
        client = CogneeApiClient("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_url_construction(self):
        client = CogneeApiClient("http://localhost:8000")
        assert client._url("/api/v1/add") == "http://localhost:8000/api/v1/add"
        assert client._url("api/v1/add") == "http://localhost:8000/api/v1/add"

    def test_custom_headers_stored(self):
        client = CogneeApiClient("http://x", headers={"X-User-Id": "abc"})
        assert client._extra_headers == {"X-User-Id": "abc"}

    def test_default_timeout(self):
        client = CogneeApiClient("http://x")
        assert client.timeout == 120.0

    def test_custom_timeout(self):
        client = CogneeApiClient("http://x", timeout=30.0)
        assert client.timeout == 30.0


class TestCogneeApiClientSharedConnection:
    """Test that a single httpx.Client is reused across calls."""

    def test_client_reused(self):
        """_get_client() should return the same object on repeated calls."""
        with patch("cognee.cli.api_client._import_httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.return_value.Client.return_value = mock_client

            api = CogneeApiClient("http://localhost:8000")
            c1 = api._get_client()
            c2 = api._get_client()
            assert c1 is c2
            # httpx.Client should only have been constructed once
            assert mock_httpx.return_value.Client.call_count == 1

    def test_close_resets_client(self):
        with patch("cognee.cli.api_client._import_httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.return_value.Client.return_value = mock_client

            api = CogneeApiClient("http://localhost:8000")
            api._get_client()
            api.close()
            assert api._client is None
            mock_client.close.assert_called_once()

    def test_context_manager(self):
        with patch("cognee.cli.api_client._import_httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.return_value.Client.return_value = mock_client

            with CogneeApiClient("http://localhost:8000") as api:
                api._get_client()
            mock_client.close.assert_called_once()


class TestRaiseForStatus:
    """Test error handling in _raise_for_status."""

    def test_success_does_not_raise(self):
        client = CogneeApiClient("http://x")
        resp = MagicMock(status_code=200)
        client._raise_for_status(resp)  # should not raise

    def test_4xx_raises_with_json_detail(self):
        client = CogneeApiClient("http://x")
        resp = MagicMock(status_code=404)
        resp.json.return_value = {"error": "not found"}
        with pytest.raises(RuntimeError, match="API error 404"):
            client._raise_for_status(resp)

    def test_5xx_raises_with_text_fallback(self):
        client = CogneeApiClient("http://x")
        resp = MagicMock(status_code=500)
        resp.json.side_effect = Exception("not json")
        resp.text = "Internal Server Error"
        with pytest.raises(RuntimeError, match="Internal Server Error"):
            client._raise_for_status(resp)


class TestAddFileDetection:
    """Test that add() detects file paths vs text strings."""

    def test_text_items_sent_as_bytes(self):
        with patch("cognee.cli.api_client._import_httpx"):
            client = CogneeApiClient("http://x")
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {"status": "ok"}
            mock_http_client = MagicMock()
            mock_http_client.post.return_value = mock_resp
            client._client = mock_http_client

            client.add(["hello world"], "ds")

            call_kwargs = mock_http_client.post.call_args
            files_arg = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
            # First file tuple: ("data", ("text_0.txt", BytesIO, "text/plain"))
            assert files_arg[0][0] == "data"
            assert files_arg[0][1][0] == "text_0.txt"
            assert files_arg[0][1][2] == "text/plain"

    def test_real_file_sent_as_upload(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("file content")
            f.flush()
            path = f.name
        try:
            with patch("cognee.cli.api_client._import_httpx"):
                client = CogneeApiClient("http://x")
                mock_resp = MagicMock(status_code=200)
                mock_resp.json.return_value = {"status": "ok"}
                mock_http_client = MagicMock()
                mock_http_client.post.return_value = mock_resp
                client._client = mock_http_client

                client.add([path], "ds")

                call_kwargs = mock_http_client.post.call_args
                files_arg = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
                assert files_arg[0][0] == "data"
                assert files_arg[0][1][0] == os.path.basename(path)
        finally:
            os.unlink(path)
