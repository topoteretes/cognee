import os
import requests
import json
from .embeddings import Embeddings
from .vector_db import VectorDB
from .response import Response


class CogneeManager:
    def __init__(self, embeddings: Embeddings = None,
                 vector_db: VectorDB = None,
                 vector_db_key: str = None,
                 embedding_api_key: str = None,
                 webhook_url: str = None,
                 lines_per_batch: int = 1000,
                 webhook_key: str = None,
                 document_id: str = None,
                 chunk_validation_url: str = None,
                 internal_api_key: str = "test123",
                 base_url="http://localhost:8000"):
        self.embeddings = embeddings if embeddings else Embeddings()
        self.vector_db = vector_db if vector_db else VectorDB()
        self.webhook_url = webhook_url
        self.lines_per_batch = lines_per_batch
        self.webhook_key = webhook_key
        self.document_id = document_id
        self.chunk_validation_url = chunk_validation_url
        self.vector_db_key = vector_db_key
        self.embeddings_api_key = embedding_api_key
        self.internal_api_key = internal_api_key
        self.base_url = base_url

    def serialize(self):
        data = {
            'EmbeddingsMetadata': json.dumps(self.embeddings.serialize()),
            'VectorDBMetadata': json.dumps(self.vector_db.serialize()),
            'WebhookURL': self.webhook_url,
            'LinesPerBatch': self.lines_per_batch,
            'DocumentID': self.document_id,
            'ChunkValidationURL': self.chunk_validation_url,
        }
        return {k: v for k, v in data.items() if v is not None}

    def upload(self, file_paths: list[str], base_url=None):
        if base_url:
            url = base_url + "/jobs"
        else:
            url = self.base_url + "/jobs"

        data = self.serialize()
        headers = self.generate_headers()
        multipart_form_data = [('file', (os.path.basename(filepath), open(filepath, 'rb'), 'application/octet-stream'))
                               for filepath in file_paths]

        print(f"embedding {len(file_paths)} documents at {url}")
        response = requests.post(url, files=multipart_form_data, headers=headers, stream=True, data=data)

        if response.status_code == 500:
            print(response.text)
            return Response(error=response.text, status_code=response.status_code)

        response_json = response.json()
        if response.status_code >= 400 and response.status_code < 500:
            print(f"Error: {response_json['error']}")

        return Response.from_json(response_json, response.status_code)

    def get_job_statuses(self, job_ids: list[int], base_url=None):
        if base_url:
            url = base_url + "/jobs/status"
        else:
            url = self.base_url + "/jobs/status"

        headers = {
            "Authorization": self.internal_api_key,
        }

        data = {
            'JobIDs': job_ids
        }

        print(f"retrieving job statuses for {len(job_ids)} jobs at {url}")
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 500:
            print(response.text)
            return Response(error=response.text, status_code=response.status_code)

        response_json = response.json()
        if response.status_code >= 400 and response.status_code < 500:
            print(f"Error: {response_json['error']}")

        return Response.from_json(response_json, response.status_code)

    def embed(self, filepath, base_url=None):
        if base_url:
            url = base_url + "/embed"
        else:
            url = self.base_url + "/embed"

        data = self.serialize()
        headers = self.generate_headers()

        files = {
            'SourceData': open(filepath, 'rb')
        }

        print(f"embedding document at file path {filepath} at {url}")
        response = requests.post(url, headers=headers, data=data, files=files)

        if response.status_code == 500:
            print(response.text)
            return Response(error=response.text, status_code=response.status_code)

        response_json = response.json()
        if response.status_code >= 400 and response.status_code < 500:
            print(f"Error: {response_json['error']}")

        return Response.from_json(response_json, response.status_code)

    def get_job_status(self, job_id, base_url=None):
        if base_url:
            url = base_url + "/jobs/" + str(job_id) + "/status"
        else:
            url = self.base_url + "/jobs/" + str(job_id) + "/status"

        headers = {
            "Authorization": self.internal_api_key,
        }

        print(f"retrieving job status for job {job_id} at {url}")
        response = requests.get(url, headers=headers)

        if response.status_code == 500:
            print(response.text)
            return Response(error=response.text, status_code=response.status_code)

        response_json = response.json()
        if response.status_code >= 400 and response.status_code < 500:
            print(f"Error: {response_json['error']}")

        return Response.from_json(response_json, response.status_code)

    def generate_headers(self):
        headers = {
            "Authorization": self.internal_api_key,
            "X-EmbeddingAPI-Key": self.embeddings_api_key,
            "X-VectorDB-Key": self.vector_db_key,
            "X-Webhook-Key": self.webhook_key
        }
        return {k: v for k, v in headers.items() if v is not None}