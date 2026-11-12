import os
import unittest
from unittest.mock import patch

from cognee.shared.utils import send_telemetry


class TestTelemetry(unittest.IsolatedAsyncioTestCase):
    @patch("cognee.shared.utils._get_api_key_tracking_id", return_value="api-key-test-id")
    @patch("cognee.shared.utils.get_persistent_id", return_value="persistent-test-id")
    @patch("cognee.shared.utils.get_anonymous_id", return_value="anonymous-test-id")
    @patch("cognee.shared.utils._telemetry_sender.enqueue")
    async def test_telemetry_enabled(
        self,
        mock_enqueue,
        _mock_get_anonymous_id,
        _mock_get_persistent_id,
        _mock_get_api_key_tracking_id,
    ):
        with patch.dict(os.environ, {"ENV": "prod"}, clear=False):
            os.environ.pop("TELEMETRY_DISABLED", None)
            send_telemetry("test_event", "test-user-id", {"test_key": "test_value"})

        mock_enqueue.assert_called_once()
        payload = mock_enqueue.call_args.args[0]

        self.assertEqual(payload["event_name"], "test_event")
        self.assertEqual(payload["user_properties"]["user_id"], "test-user-id")
        self.assertEqual(payload["properties"]["user_id"], "test-user-id")
        self.assertEqual(payload["properties"]["test_key"], "test_value")
        self.assertEqual(payload["properties"]["anonymous_id"], "anonymous-test-id")
        self.assertEqual(payload["properties"]["persistent_id"], "persistent-test-id")

    @patch("cognee.shared.utils._telemetry_sender.enqueue")
    async def test_telemetry_disabled(self, mock_enqueue):
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            send_telemetry("disabled_test", "user123", {"key": "value"})

        mock_enqueue.assert_not_called()

    @patch("cognee.shared.utils._telemetry_sender.enqueue")
    async def test_telemetry_dev_env(self, mock_enqueue):
        with patch.dict(os.environ, {"ENV": "dev"}, clear=False):
            os.environ.pop("TELEMETRY_DISABLED", None)
            send_telemetry("dev_test", "user123", {"key": "value"})

        mock_enqueue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
