import unittest
import subprocess
import time
import os
import signal
import requests
import sys


class TestCogneeServerStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the Cognee server - just check if the server can start without errors
        cls.server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "cognee.api.client:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        # Give the server some time to start
        time.sleep(20)

        # Check if server started with errors
        if cls.server_process.poll() is not None:
            stderr = cls.server_process.stderr.read().decode("utf-8")
            print(f"Server failed to start: {stderr}", file=sys.stderr)
            raise RuntimeError(f"Server failed to start: {stderr}")

    @classmethod
    def tearDownClass(cls):
        # Terminate the server process
        if hasattr(cls, "server_process") and cls.server_process:
            os.killpg(os.getpgid(cls.server_process.pid), signal.SIGTERM)
            cls.server_process.wait()

    def test_server_is_running(self):
        """Test that the server is running and can accept connections."""
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health", timeout=10)
        self.assertEqual(health_response.status_code, 200)

        # Test root endpoint
        root_response = requests.get("http://localhost:8000/", timeout=10)
        self.assertEqual(root_response.status_code, 200)
        self.assertIn("message", root_response.json())
        self.assertEqual(root_response.json()["message"], "Hello, World, I am alive!")


if __name__ == "__main__":
    unittest.main()
