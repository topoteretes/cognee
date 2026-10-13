"""Retry a typed capture after an uncertain response without creating another entry.

Set COGNEE_BASE_URL and COGNEE_API_KEY for a server advertising
capture_idempotency_v1 (SQL session cache). This creates one isolated example
session; persist client_event_id with your event before the first request.
"""

import json
import os
import urllib.request
from uuid import uuid4


def main():
    base = os.environ["COGNEE_BASE_URL"].rstrip("/")
    headers = {"Content-Type": "application/json", "X-Api-Key": os.environ["COGNEE_API_KEY"]}
    event_id = str(uuid4())
    body = {
        "client_event_id": event_id,
        "dataset_name": "capture_idempotency_example",
        "session_id": "capture_example_" + str(uuid4()),
        "entry": {"type": "qa", "question": "Which queue?", "answer": "Use a durable spool."},
    }

    def send():
        request = urllib.request.Request(
            base + "/api/v1/remember/entry", data=json.dumps(body).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)

    first = send()
    replay = send()  # Same request after an imagined lost acknowledgement.
    assert replay == first
    assert replay["client_event_id"] == event_id
    print("Stable receipt:", replay["entry_id"])


if __name__ == "__main__":
    main()
