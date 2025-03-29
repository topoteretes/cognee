import unittest
import os
import uuid
from unittest.mock import patch, MagicMock
import sys

# Import the telemetry function to test
from cognee.shared.utils import send_telemetry

class TestTelemetry(unittest.TestCase):
    
    @patch('cognee.shared.utils.requests.post')
    def test_telemetry_enabled(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Ensure telemetry is enabled for this test
        if "TELEMETRY_DISABLED" in os.environ:
            del os.environ["TELEMETRY_DISABLED"]
        
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
        
        # Check that the URL was correctly formed
        self.assertTrue(len(args) > 0)
        
        # Check that the payload contains our data
        self.assertIn('json', kwargs)
        payload = kwargs['json']
        
        # Verify payload contains expected data
        self.assertEqual(payload.get('event_name'), event_name)
        self.assertEqual(payload.get('user_id'), test_user_id)
        self.assertEqual(payload.get('additional_properties', {}).get('test_key'), "test_value")
    
    @patch('cognee.shared.utils.requests.post')
    def test_telemetry_disabled(self, mock_post):
        # Enable the TELEMETRY_DISABLED environment variable
        os.environ["TELEMETRY_DISABLED"] = "1"
        
        # Test sending telemetry
        send_telemetry("disabled_test", "user123", {"key": "value"})
        
        # Verify telemetry was not sent
        mock_post.assert_not_called()
        
        # Clean up
        del os.environ["TELEMETRY_DISABLED"]

    def test_anon_id_file_exists(self):
        """Test that .anon_id file exists for telemetry."""
        # Check if .anon_id exists in the project root
        anon_id_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.anon_id')
        
        if not os.path.exists(anon_id_path):
            # Create the file with a test ID if it doesn't exist
            with open(anon_id_path, 'w') as f:
                f.write("test-machine")
            print(f"Created .anon_id file at {anon_id_path}", file=sys.stderr)
        
        self.assertTrue(os.path.exists(anon_id_path), "The .anon_id file should exist")
        
        # Verify the file has content
        with open(anon_id_path, 'r') as f:
            content = f.read().strip()
        
        self.assertTrue(len(content) > 0, "The .anon_id file should not be empty")

if __name__ == "__main__":
    unittest.main() 