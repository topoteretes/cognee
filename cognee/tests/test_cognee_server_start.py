import unittest
import subprocess
import time
import os
import signal
import requests
from pathlib import Path
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
        time.sleep(35)

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
        health_response = requests.get("http://localhost:8000/health", timeout=15)
        self.assertEqual(health_response.status_code, 200)

        # Test root endpoint
        root_response = requests.get("http://localhost:8000/", timeout=15)
        self.assertEqual(root_response.status_code, 200)
        self.assertIn("message", root_response.json())
        self.assertEqual(root_response.json()["message"], "Hello, World, I am alive!")

        # Login request
        url = "http://127.0.0.1:8000/api/v1/auth/login"
        form_data = {
            "username": "default_user@example.com",
            "password": "default_password",
        }
        login_response = requests.post(url, data=form_data, timeout=15)
        login_response.raise_for_status()  # raises on HTTP 4xx/5xx

        # Define Bearer token to use for authorization
        auth_var = (
            "Bearer " + login_response.json()["access_token"]
        )  # e.g. "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6..."

        # Add request
        url = "http://127.0.0.1:8000/api/v1/add"
        file_path = Path(os.path.join(Path(__file__).parent, "test_data/example.png"))
        headers = {"Authorization": auth_var}

        form_data = {"datasetName": "test"}

        file = {
            "data": (
                file_path.name,
                open(file_path, "rb"),
            )
        }

        add_response = requests.post(url, headers=headers, data=form_data, files=file, timeout=50)
        add_response.raise_for_status()  # raise if HTTP 4xx/5xx

        # Cognify request
        url = "http://127.0.0.1:8000/api/v1/cognify"
        headers = {
            "Authorization": auth_var,
            "Content-Type": "application/json",
        }

        payload = {"datasets": ["test"]}

        cognify_response = requests.post(url, headers=headers, json=payload, timeout=150)
        cognify_response.raise_for_status()  # raises on HTTP 4xx/5xx

        # TODO: Add test to verify cognify pipeline is complete before testing search

        # Search request
        url = "http://127.0.0.1:8000/api/v1/search"

        headers = {
            "Authorization": auth_var,
            "Content-Type": "application/json",
        }

        payload = {"searchType": "GRAPH_COMPLETION", "query": "What's in the document?"}

        search_response = requests.post(url, headers=headers, json=payload, timeout=50)
        search_response.raise_for_status()  # raises on HTTP 4xx/5xx


if __name__ == "__main__":
    unittest.main()
