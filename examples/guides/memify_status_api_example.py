import json
from urllib import request


BASE_URL = "https://your-cognee-host.example"


def post_json(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req) as response:
        return json.load(response)


def get_json(path: str) -> dict:
    with request.urlopen(f"{BASE_URL}{path}") as response:
        return json.load(response)


def main():
    queued_run = post_json(
        "/api/v1/memify",
        {
            "dataset_name": "research_notes",
            "run_in_background": True,
        },
    )
    print("Queued memify run:")
    print(json.dumps(queued_run, indent=2))

    pipeline_run_id = queued_run["pipeline_run_id"]

    run_status = get_json(f"/api/v1/memify/status/{pipeline_run_id}")
    print("Status for exact run:")
    print(json.dumps(run_status, indent=2))

    latest_dataset_status = get_json("/api/v1/memify/status?dataset_name=research_notes")
    print("Latest memify run for dataset:")
    print(json.dumps(latest_dataset_status, indent=2))


if __name__ == "__main__":
    main()