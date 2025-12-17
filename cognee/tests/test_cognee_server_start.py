import unittest
import subprocess
import time
import os
import signal
import requests
from pathlib import Path
import sys
import uuid
import json


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
            if hasattr(os, "killpg"):
                # Unix-like systems: Use process groups
                os.killpg(os.getpgid(cls.server_process.pid), signal.SIGTERM)
            else:
                # Windows: Just terminate the main process
                cls.server_process.terminate()
            cls.server_process.wait()

    def test_server_is_running(self):
        """Test that the server is running and can accept connections."""
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health", timeout=15)
        self.assertIn(health_response.status_code, [200])

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

        dataset_name = f"test_{uuid.uuid4().hex[:8]}"
        form_data = {"datasetName": dataset_name}

        file = {
            "data": (
                file_path.name,
                open(file_path, "rb"),
            )
        }

        ontology_key = f"test_ontology_{uuid.uuid4().hex[:8]}"
        payload = {"datasets": [dataset_name], "ontology_key": [ontology_key]}

        add_response = requests.post(url, headers=headers, data=form_data, files=file, timeout=50)
        if add_response.status_code not in [200, 201]:
            add_response.raise_for_status()

        ontology_content = b"""<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns="http://example.org/ontology#"
         xml:base="http://example.org/ontology">

                <owl:Ontology rdf:about="http://example.org/ontology"/>

                <!-- Classes -->
                <owl:Class rdf:ID="Problem"/>
                <owl:Class rdf:ID="HardwareProblem"/>
                <owl:Class rdf:ID="SoftwareProblem"/>
                <owl:Class rdf:ID="Concept"/>
                <owl:Class rdf:ID="Object"/>
                <owl:Class rdf:ID="Joke"/>
                <owl:Class rdf:ID="Image"/>
                <owl:Class rdf:ID="Person"/>

                <rdf:Description rdf:about="#HardwareProblem">
                    <rdfs:subClassOf rdf:resource="#Problem"/>
                    <rdfs:comment>A failure caused by physical components.</rdfs:comment>
                </rdf:Description>

                <rdf:Description rdf:about="#SoftwareProblem">
                    <rdfs:subClassOf rdf:resource="#Problem"/>
                    <rdfs:comment>An error caused by software logic or configuration.</rdfs:comment>
                </rdf:Description>

                <rdf:Description rdf:about="#Person">
                    <rdfs:comment>A human being or individual.</rdfs:comment>
                </rdf:Description>

                <!-- Individuals -->
                <Person rdf:ID="programmers">
                    <rdfs:label>Programmers</rdfs:label>
                </Person>

                <Object rdf:ID="light_bulb">
                    <rdfs:label>Light Bulb</rdfs:label>
                </Object>

                <HardwareProblem rdf:ID="hardware_problem">
                    <rdfs:label>Hardware Problem</rdfs:label>
                </HardwareProblem>

            </rdf:RDF>"""

        ontology_response = requests.post(
            "http://127.0.0.1:8000/api/v1/ontologies",
            headers=headers,
            files=[("ontology_file", ("test.owl", ontology_content, "application/xml"))],
            data={
                "ontology_key": ontology_key,
                "description": "Test ontology",
            },
        )
        self.assertEqual(ontology_response.status_code, 200)

        # Cognify request
        url = "http://127.0.0.1:8000/api/v1/cognify"
        headers = {
            "Authorization": auth_var,
            "Content-Type": "application/json",
        }

        cognify_response = requests.post(url, headers=headers, json=payload, timeout=150)
        if cognify_response.status_code not in [200, 201]:
            cognify_response.raise_for_status()

        datasets_response = requests.get("http://127.0.0.1:8000/api/v1/datasets", headers=headers)

        datasets = datasets_response.json()
        dataset_id = None
        for dataset in datasets:
            if dataset["name"] == dataset_name:
                dataset_id = dataset["id"]
                break

        graph_response = requests.get(
            f"http://127.0.0.1:8000/api/v1/datasets/{dataset_id}/graph", headers=headers
        )
        self.assertEqual(graph_response.status_code, 200)

        graph_data = graph_response.json()
        ontology_nodes = [
            node for node in graph_data.get("nodes") if node.get("properties").get("ontology_valid")
        ]

        self.assertGreater(
            len(ontology_nodes), 0, "No ontology nodes found - ontology was not integrated"
        )

        # TODO: Add test to verify cognify pipeline is complete before testing search

        # Search request
        url = "http://127.0.0.1:8000/api/v1/search"

        headers = {
            "Authorization": auth_var,
            "Content-Type": "application/json",
        }

        payload = {"searchType": "GRAPH_COMPLETION", "query": "What's in the document?"}

        search_response = requests.post(url, headers=headers, json=payload, timeout=50)
        if search_response.status_code not in [200, 201]:
            search_response.raise_for_status()


if __name__ == "__main__":
    unittest.main()
