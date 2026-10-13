"""Inspect an authorized recall index and expand up to five citations.

COGNEE_BASE_URL and COGNEE_API_KEY select your running server. Run with a
query and dataset UUID: python examples/guides/recall_evidence.py 'policy' UUID
"""

import json
import os
import sys
import urllib.request


def post(path: str, payload: dict):
    request = urllib.request.Request(
        os.environ["COGNEE_BASE_URL"].rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"X-Api-Key": os.environ["COGNEE_API_KEY"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def main():
    index = post(
        "/api/v1/recall",
        {
            "query": sys.argv[1],
            "dataset_ids": [sys.argv[2]],
            "scope": ["graph"],
            "evidence_mode": "index",
            "top_k": 5,
        },
    )
    print(json.dumps(index, indent=2))
    references = [item["reference"] for item in index["items"] if item["status"] == "available"][:5]
    if references:
        print(json.dumps(post("/api/v1/recall/evidence", {"references": references}), indent=2))


if __name__ == "__main__":
    main()
