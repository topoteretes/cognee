import unittest
import os
import uuid
from unittest.mock import patch, MagicMock
import sys

# Import the telemetry function to test
from cognee.shared.utils import send_telemetry


class TestTelemetry(unittest.TestCase):
    @patch("cognee.shared.utils.requests.post")
    def test_telemetry_enabled(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Check if .anon_id exists in the project root
        anon_id_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".anon_id"
        )

        if not os.path.exists(anon_id_path):
            # Create the file with a test ID if it doesn't exist
            with open(anon_id_path, "w") as f:
                f.write("test-machine")
            print(f"Created .anon_id file at {anon_id_path}", file=sys.stderr)

        self.assertTrue(os.path.exists(anon_id_path), "The .anon_id file should exist")

        # Verify the file has content
        with open(anon_id_path, "r") as f:
            content = f.read().strip()

        self.assertTrue(len(content) > 0, "The .anon_id file should not be empty")

        # Ensure telemetry is enabled for this test
        if "TELEMETRY_DISABLED" in os.environ:
            del os.environ["TELEMETRY_DISABLED"]

        # Make sure ENV is not test or dev
        original_env = os.environ.get("ENV")
        os.environ["ENV"] = "prod"  # Set to dev to ensure telemetry is sent

        # Generate a random user ID for testing
        test_user_id = str(uuid.uuid4())

        # Test sending telemetry
        event_name = "test_event"
        additional_props = {"test_key": "test_value"}

        send_telemetry(event_name, test_user_id, additional_props)

        # Verify telemetry was sent
        mock_post.assert_called_once()

        # Get the args that were passed to post
        args, kwargs = mock_post.call_args

        # Check that the payload contains our data
        self.assertIn("json", kwargs)
        payload = kwargs["json"]

        # Verify payload contains expected data
        self.assertEqual(payload.get("event_name"), event_name)

        # Check that user_id is in the correct nested structure
        self.assertIn("user_properties", payload)
        self.assertEqual(payload["user_properties"].get("user_id"), str(test_user_id))

        # Also check that user_id is in the properties
        self.assertIn("properties", payload)
        self.assertEqual(payload["properties"].get("user_id"), str(test_user_id))

        # Check that additional properties are included
        self.assertEqual(payload["properties"].get("test_key"), "test_value")

        # Restore original ENV if it existed
        if original_env is not None:
            os.environ["ENV"] = original_env
        else:
            del os.environ["ENV"]

    @patch("cognee.shared.utils.requests.post")
    def test_telemetry_disabled(self, mock_post):
        # Enable the TELEMETRY_DISABLED environment variable
        os.environ["TELEMETRY_DISABLED"] = "1"

        # Test sending telemetry
        send_telemetry("disabled_test", "user123", {"key": "value"})

        # Verify telemetry was not sent
        mock_post.assert_not_called()

        # Clean up
        del os.environ["TELEMETRY_DISABLED"]

    @patch("cognee.shared.utils.requests.post")
    def test_telemetry_dev_env(self, mock_post):
        # Set ENV to dev which should disable telemetry
        original_env = os.environ.get("ENV")
        os.environ["ENV"] = "dev"

        if "TELEMETRY_DISABLED" in os.environ:
            del os.environ["TELEMETRY_DISABLED"]

        # Test sending telemetry
        send_telemetry("dev_test", "user123", {"key": "value"})

        # Verify telemetry was not sent in dev environment
        mock_post.assert_not_called()

        # Restore original ENV if it existed
        if original_env is not None:
            os.environ["ENV"] = original_env
        else:
            del os.environ["ENV"]


if __name__ == "__main__":
    unittest.main()
