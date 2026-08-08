import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from cognee.shared.utils import send_telemetry


class TestTelemetry(unittest.IsolatedAsyncioTestCase):
    @patch("cognee.shared.utils._get_api_key_tracking_id", return_value="api-key-test-id")
    @patch("cognee.shared.utils.get_persistent_id", return_value="persistent-test-id")
    @patch("cognee.shared.utils.get_anonymous_id", return_value="anonymous-test-id")
    @patch("cognee.shared.utils._send_telemetry_request", new_callable=AsyncMock)
    async def test_telemetry_enabled(
        self,
        mock_send_telemetry_request,
        _mock_get_anonymous_id,
        _mock_get_persistent_id,
        _mock_get_api_key_tracking_id,
    ):
        request_started = asyncio.Event()

        async def capture_request(_payload):
            request_started.set()

        mock_send_telemetry_request.side_effect = capture_request

        with patch.dict(os.environ, {"ENV": "prod"}, clear=False):
            os.environ.pop("TELEMETRY_DISABLED", None)
            send_telemetry("test_event", "test-user-id", {"test_key": "test_value"})
            await asyncio.wait_for(request_started.wait(), timeout=1)

        mock_send_telemetry_request.assert_awaited_once()
        payload = mock_send_telemetry_request.await_args.args[0]

        self.assertEqual(payload["event_name"], "test_event")
        self.assertEqual(payload["user_properties"]["user_id"], "test-user-id")
        self.assertEqual(payload["properties"]["user_id"], "test-user-id")
        self.assertEqual(payload["properties"]["test_key"], "test_value")
        self.assertEqual(payload["properties"]["anonymous_id"], "anonymous-test-id")
        self.assertEqual(payload["properties"]["persistent_id"], "persistent-test-id")

    @patch("cognee.shared.utils._send_telemetry_request", new_callable=AsyncMock)
    async def test_telemetry_disabled(self, mock_send_telemetry_request):
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            send_telemetry("disabled_test", "user123", {"key": "value"})

        mock_send_telemetry_request.assert_not_called()

    @patch("cognee.shared.utils._send_telemetry_request", new_callable=AsyncMock)
    async def test_telemetry_dev_env(self, mock_send_telemetry_request):
        with patch.dict(os.environ, {"ENV": "dev"}, clear=False):
            os.environ.pop("TELEMETRY_DISABLED", None)
            send_telemetry("dev_test", "user123", {"key": "value"})

        mock_send_telemetry_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
