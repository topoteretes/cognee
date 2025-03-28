import unittest
import subprocess
import time
import os
import signal
import requests
from pathlib import Path

class TestCogneeServerStart(unittest.TestCase):
    def setUp(self):
        # Start the Cognee server
        self.server_process = subprocess.Popen(
            ["poetry", "run", "uvicorn", "cognee.api.client:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        # Give the server some time to start
        time.sleep(5)

    def tearDown(self):
        # Terminate the server process
        if hasattr(self, 'server_process') and self.server_process:
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            self.server_process.wait()

    def test_server_is_running(self):
        """Test that the server is running and responding to requests."""
        try:
            # Test health endpoint
            health_response = requests.get("http://localhost:8000/health")
            self.assertEqual(health_response.status_code, 200)
            
            # Test root endpoint
            root_response = requests.get("http://localhost:8000/")
            self.assertEqual(root_response.status_code, 200)
            self.assertIn("message", root_response.json())
            self.assertEqual(root_response.json()["message"], "Hello, World, I am alive!")
        except requests.RequestException as e:
            self.fail(f"Server is not running: {e}")

if __name__ == "__main__":
    unittest.main() 